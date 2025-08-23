"""Headroom Pretotype demo app — v1 pilot context: Germany (DE), oncology brands."""
import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Headroom Pretotype", layout="wide")

st.title("Headroom Analysis Pretotype")
st.caption(f"Quick demo app using mock CSV data. Adjust weights in config/scoring.yaml · Data as of {date.today().isoformat()}")

# Ensure sibling 'src' package is importable when running via Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loaders import load_public_data
from src.compute import compute_metrics
from src.rules import recommend_actions
from src.utils import load_yaml
from src.llm_recs import llm_recommend_actions

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Cache wrapper for LLM calls keyed by the JSON context
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_llm_call(context_key: str):
    try:
        ctx = json.loads(context_key)
    except Exception:
        ctx = {}
    return llm_recommend_actions(None, ctx)

@st.cache_data
def _load():
    ds_cfg = load_yaml(os.path.join(BASE_DIR, "config", "datasources.yaml"))
    sc_cfg = load_yaml(os.path.join(BASE_DIR, "config", "scoring.yaml"))
    data = load_public_data(ds_cfg, base_dir=BASE_DIR)
    return data, sc_cfg

with st.sidebar:
    st.header("Controls")
    data, scoring_cfg = _load()
    # Filters for market/brand/quarter based on loaded markets table
    unique_markets = sorted(data["markets"]["market"].unique())
    sel_market = st.selectbox("Market", unique_markets)
    brands = sorted(data["markets"][data["markets"]["market"] == sel_market]["brand"].unique())
    sel_brand = st.selectbox("Brand", brands)
    quarters_raw = data["markets"][ (data["markets"]["market"] == sel_market) & (data["markets"]["brand"] == sel_brand) ]["quarter"].unique()
    # Sort quarters descending (expects format like YYYYQx)
    try:
        quarters = sorted(list(quarters_raw), key=lambda q: (int(str(q)[:4]), int(str(q)[-1])), reverse=True)
    except Exception:
        quarters = sorted(list(quarters_raw), reverse=True)
    sel_quarter = st.selectbox("Quarter", quarters)
    apply_filters = st.checkbox("Apply filters to views", value=True)
    st.divider()
    st.subheader("Help & Docs")
    try:
        bl_path = os.path.join(BASE_DIR, "docs", "business_logic.md")
        du_path = os.path.join(BASE_DIR, "docs", "data_usage.md")
        gl_path = os.path.join(BASE_DIR, "docs", "glossary.md")
        with open(bl_path, "r", encoding="utf-8") as f:
            bl_md = f.read()
        with open(du_path, "r", encoding="utf-8") as f:
            du_md = f.read()
        with open(gl_path, "r", encoding="utf-8") as f:
            gl_md = f.read()
        with st.expander("Business Logic"):
            st.markdown(bl_md)
        with st.expander("Data Usage"):
            st.markdown(du_md)
        with st.expander("Glossary"):
            st.markdown(gl_md)
    except Exception:
        st.caption("Docs not available.")

metrics = compute_metrics(data["markets"], data["actuals"], data["drivers"], scoring_cfg=scoring_cfg)
# Consistent naming across UI: use Priority_Score everywhere
if "ROI_Score" in metrics.columns:
    metrics = metrics.rename(columns={"ROI_Score": "Priority_Score"})

def _resolve_units_per_patient(sc_cfg: dict, market: str, brand: str) -> float:
    """Resolve units-per-patient conversion with precedence:
    1) assumptions.units_per_patient_equiv_by.market_brand["{market}::{brand}"]
    2) assumptions.units_per_patient_equiv_by.brand[brand]
    3) assumptions.units_per_patient_equiv_by.market[market]
    4) assumptions.units_per_patient_equiv (global default, default=1.0)

    Also supports nested map: assumptions.units_per_patient_equiv_by[market][brand].
    """
    assumptions = (sc_cfg or {}).get("assumptions", {})
    default = float(assumptions.get("units_per_patient_equiv", 1.0))
    by = assumptions.get("units_per_patient_equiv_by", {}) or {}
    # exact market_brand key
    try:
        mb = by.get("market_brand", {}) or {}
        key = f"{market}::{brand}"
        if key in mb:
            return float(mb[key])
    except Exception:
        pass
    # nested structure by market then brand
    try:
        nested = by.get("by_market", {}) or {}
        if market in nested:
            if isinstance(nested[market], dict) and brand in nested[market]:
                return float(nested[market][brand])
    except Exception:
        pass
    # brand-only override
    try:
        bmap = by.get("brand", {}) or {}
        if brand in bmap:
            return float(bmap[brand])
    except Exception:
        pass
    # market-only override
    try:
        mmap = by.get("market", {}) or {}
        if market in mmap:
            return float(mmap[market])
    except Exception:
        pass
    return default

# Resolve per selection
units_per_patient = _resolve_units_per_patient(scoring_cfg, sel_market, sel_brand)

# Quarter sorting helper (robust)
def _q_key(q):
    s = str(q).strip().upper().replace(" ", "")
    year = int(s[:4])
    qtr = int(s[-1])
    return (year, qtr)

# Data-quality surface: warn and exclude missing rows from main ranking by default
dq_missing_count = int(metrics["DQ_Missing"].sum()) if "DQ_Missing" in metrics.columns else 0
if dq_missing_count > 0:
    st.warning(f"{dq_missing_count} row(s) have missing critical inputs and are excluded from the ranking table. Fix inputs or override once validated.")

# Build a filtered view for downstream visualizations if enabled
try:
    if apply_filters:
        view = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand) & (metrics["quarter"] == sel_quarter)]
    else:
        view = metrics
except NameError:
    # Fallback in case sidebar hasn't set apply_filters yet
    view = metrics

# Top-of-page: Funnel – sets the story context first
row_top = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand) & (metrics["quarter"] == sel_quarter)]
if row_top.empty:
    st.info("No data for selection.")
else:
    rt = row_top.iloc[0]
    # Side-by-side layout: Funnel (left) and Headroom (right)
    col_funnel, col_headroom = st.columns(2)

    # Left: Funnel – Patients to Units
    with col_funnel:
        st.subheader("Funnel – Patients to Units")
        patients_total = float(rt["patients_total"])
        patients_eligible = float(rt["patients_eligible"])
        units_sold = float(rt["units_sold"])
        # Units conversion factor (guarded) used consistently across visuals
        try:
            units_per_patient = float(rt.get("units_per_patient", 1.0))
        except Exception:
            units_per_patient = 1.0

        funnel_stages = [
            "Total Patient Population",
            "Eligible Patients",
            "Treated Patients (brand)",
        ]
        funnel_values = [patients_total, patients_eligible, units_sold]
        hover_text = [
            "Epidemiology base (incidence/prevalence)",
            "Meet label/guideline criteria for the brand",
            "On selected brand’s therapy in the period (proxied by that brand’s units sold)",
        ]

        fig_top = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textposition="inside",
            textinfo="value+percent initial",
            hoverinfo="text+name",
            textfont=dict(color="white"),
            hovertext=hover_text,
            marker={"color": ["#4C78A8", "#72B7B2", "#54A24B"]},
        ))
        fig_top.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=10))
        fig_top.update_xaxes(automargin=True)
        fig_top.update_yaxes(automargin=True)
        st.plotly_chart(fig_top, use_container_width=True)
        # Place concise conversion note under the Funnel
        st.caption(f"Units conversion: patients × {units_per_patient:.2f} = units. 'Treated Patients (brand)' reflects units sold.")
        # Explanation drop bar (expander) under Funnel
        with st.expander("Funnel definitions"):
            st.markdown(
                """
                - Total Patient Population: epidemiology base (incidence/prevalence)
                - Eligible Patients: meet label/guideline criteria for the brand
                - Treated Patients (brand): patients on selected brand in period (proxied by units)
                - Units are patients × conversion factor
                """
            )

    # Right: Headroom – Today (units)
    with col_headroom:
        st.subheader("Headroom – Today (units)")
        # Pull inputs safely
        pot_val = rt.get("Potential_Units")
        act_val = rt.get("units_sold")
        hdr_val = rt.get("Headroom_Units")
        price_val = rt.get("price_per_unit")
        # Validate essentials
        if any(v is None for v in [pot_val, act_val, hdr_val]):
            st.info("Headroom waterfall unavailable (missing Potential/Actual/Headroom).")
        else:
            pot = float(pot_val) if pd.notna(pot_val) else 0.0
            act = float(act_val) if pd.notna(act_val) else 0.0
            hdr_u = float(hdr_val) if pd.notna(hdr_val) else max(pot - act, 0.0)
            price = float(price_val) if (price_val is not None and pd.notna(price_val)) else 0.0
            hdr_v = hdr_u * price

            fig_hw = go.Figure(go.Waterfall(
                measure=["relative", "relative", "total"],
                x=["Actual (u)", "Headroom (u)", "Total Potential (u)"],
                y=[act, hdr_u, pot],
                text=[f"{act:,.0f}", f"{hdr_u:,.0f}", f"{pot:,.0f}"],
                textposition="inside",
                connector={"line": {"color": "#AAA"}},
                # Use Waterfall color APIs: increasing/decreasing/totals
                increasing={"marker": {"color": "#54A24B"}},   # Actual & Headroom (positive) both green to match Treated Patients
                decreasing={"marker": {"color": "#E45756"}},   # if any negative
                totals={"marker": {"color": "#1F77B4"}},       # Total Potential
            ))
            fig_hw.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=10, b=10),
                yaxis=dict(rangemode="tozero"),
                xaxis=dict(categoryorder="array", categoryarray=["Actual (u)", "Headroom (u)", "Total Potential (u)"]),
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )
            # Hide axis descriptions on Headroom chart for a cleaner look
            fig_hw.update_xaxes(automargin=True, showticklabels=False, ticks="")
            fig_hw.update_yaxes(automargin=True, showticklabels=False, showgrid=False, zeroline=False)
            st.plotly_chart(fig_hw, use_container_width=True)
            # Place concise conversion note under the Headroom waterfall
            st.caption(f"All bars in units. Same conversion factor applied: {units_per_patient:.2f} units per patient.")
            # Explanation drop bar (expander) under Headroom
            with st.expander("Headroom definitions"):
                st.markdown(
                    """
                    - Actual (u): units sold in the selected period
                    - Headroom (u): Potential − Actual (clipped at 0)
                    - Total Potential (u): eligible × adoption ceiling converted to units
                    - Composition: Actual + Headroom = Total Potential
                    """
                )
            # KPIs under chart to preserve same chart width
            try:
                currency = (scoring_cfg.get("formatting", {}) if 'scoring_cfg' in globals() else {}).get("headroom_currency", "EUR")
            except Exception:
                currency = "EUR"
            sales_value_top = float(rt.get('units_sold', 0.0)) * float(rt.get('price_per_unit', 0.0))
            headroom_value_top = float(rt.get('Headroom_Value', 0.0))
            total_potential_value = sales_value_top + headroom_value_top
            growth_pct = (headroom_value_top / total_potential_value * 100.0) if total_potential_value > 0 else 0.0
            kL, kM, kR = st.columns(3)
            with kL:
                st.metric(f"Sales Value ({currency})", f"{sales_value_top:,.0f}")
            with kM:
                st.metric("Growth Potential (%)", f"{growth_pct:.1f}%")
            with kR:
                st.metric(f"Headroom ({currency})", f"{headroom_value_top:,.0f}")

    st.subheader("ΔSales – Last Period (€)")
    # ΔSales last period (Market + Share + Price + Residual) — centered under Headroom
    cL3, cM3, cR3 = st.columns([1, 2, 1])
    with cM3:
        try:
            brand_hist = metrics[metrics["market"].eq(sel_market)].copy()
            bh = brand_hist[brand_hist["brand"].eq(sel_brand)].copy()
            if len(bh) >= 1:
                # sort quarters safely
                bh["_q_sort"] = bh["quarter"].astype(str).apply(_q_key)
                bh = bh.sort_values("_q_sort")
                quarters_list = bh["quarter"].tolist()
                # robust compare via normalized strings
                q_norm = [str(q).strip().upper().replace(" ", "") for q in quarters_list]
                sel_norm = str(sel_quarter).strip().upper().replace(" ", "")
                q_t = quarters_list[-1] if sel_norm not in q_norm else quarters_list[q_norm.index(sel_norm)]
                idx_t = quarters_list.index(q_t)
                if idx_t == 0:
                    # auto-fallback to earliest adjacent pair to avoid forcing user to pick a different quarter
                    if len(quarters_list) >= 2:
                        q_tm1 = quarters_list[0]
                        q_t = quarters_list[1]
                        b_t = bh[bh["quarter"].eq(q_t)].iloc[0]
                        b_tm1 = bh[bh["quarter"].eq(q_tm1)].iloc[0]
                        fallback_used = True
                    else:
                        st.info("ΔSales waterfall unavailable (no prior period for this brand).")
                        fallback_used = False
                else:
                    q_tm1 = quarters_list[idx_t - 1]
                    b_t = bh[bh["quarter"].eq(q_t)].iloc[0]
                    b_tm1 = bh[bh["quarter"].eq(q_tm1)].iloc[0]
                    fallback_used = False
                # Helper prices
                def _price(row):
                    try:
                        if "net_sales" in row.index and pd.notna(row["net_sales"]) and row["units_sold"] not in (0, None):
                            return float(row["net_sales"]) / float(row["units_sold"]) if float(row["units_sold"]) != 0 else float(row.get("price_per_unit", 0.0))
                    except Exception:
                        pass
                    return float(row.get("price_per_unit", 0.0))
                Units_b_t = float(b_t["units_sold"]); Units_b_tm1 = float(b_tm1["units_sold"])
                Price_b_t = _price(b_t); Price_b_tm1 = _price(b_tm1)
                # Prefer net_sales for truth if present; fallback to units×price
                Sales_b_t = (float(b_t["net_sales"]) if ("net_sales" in b_t.index and pd.notna(b_t.get("net_sales", None))) else Units_b_t * Price_b_t)
                Sales_b_tm1 = (float(b_tm1["net_sales"]) if ("net_sales" in b_tm1.index and pd.notna(b_tm1.get("net_sales", None))) else Units_b_tm1 * Price_b_tm1)
                mkt = metrics[metrics["market"].eq(sel_market)]
                Units_m_t = float(mkt[mkt["quarter"].eq(q_t)]["units_sold"].sum())
                Units_m_tm1 = float(mkt[mkt["quarter"].eq(q_tm1)]["units_sold"].sum())
                # Effects (simple illustrative decomposition)
                try:
                    Share_t = Units_b_t / Units_m_t if Units_m_t else 0.0
                    Share_tm1 = Units_b_tm1 / Units_m_tm1 if Units_m_tm1 else 0.0
                except Exception:
                    Share_t, Share_tm1 = 0.0, 0.0
                Market_Effect = (Units_m_t - Units_m_tm1) * Price_b_tm1 * Share_tm1
                Share_Effect = (Share_t - Share_tm1) * Units_m_t * Price_b_tm1
                Price_Effect = (Price_b_t - Price_b_tm1) * Units_b_t
                Delta_Sales = Sales_b_t - Sales_b_tm1
                Residual = Delta_Sales - (Market_Effect + Share_Effect + Price_Effect)
                fig_ds = go.Figure(go.Waterfall(
                    measure=["relative", "relative", "relative", "relative", "total"],
                    x=["Market", "Share", "Price", "Residual", "ΔSales"],
                    y=[Market_Effect, Share_Effect, Price_Effect, Residual, Delta_Sales],
                    text=[f"€{Market_Effect:,.0f}", f"€{Share_Effect:,.0f}", f"€{Price_Effect:,.0f}", f"€{Residual:,.0f}", f"€{Delta_Sales:,.0f}"],
                    textposition="inside",
                    connector={"line": {"color": "#AAA"}},
                ))
                fig_ds.update_layout(
                    height=420,
                    width=800,
                    margin=dict(l=20, r=20, t=30, b=30),
                    yaxis=dict(rangemode="tozero"),
                    xaxis=dict(categoryorder="array", categoryarray=["Market", "Share", "Price", "Residual", "ΔSales"]),
                    uniformtext_minsize=10,
                    uniformtext_mode="hide",
                )
                fig_ds.update_xaxes(tickangle=0, automargin=True)
                fig_ds.update_yaxes(automargin=True)
                st.plotly_chart(fig_ds, use_container_width=True)
                price_basis = "net" if ("net_sales" in b_t.index and pd.notna(b_t.get("net_sales", None)) and b_t["units_sold"]) else "list"
                cap = f"Decomposition: {q_tm1} → {q_t} · Price basis: {price_basis}"
                if idx_t == 0 and fallback_used:
                    cap += " · Note: selected quarter has no prior; showing earliest available pair."
                st.caption(cap)
                with st.expander("ΔSales calc details"):
                    st.write({
                        "Sales_t (€)": f"{Sales_b_t:,.0f}",
                        "Sales_t-1 (€)": f"{Sales_b_tm1:,.0f}",
                        "ΔSales (€)": f"{Delta_Sales:,.0f}",
                        "Units_t": f"{Units_b_t:,.0f}",
                        "Units_t-1": f"{Units_b_tm1:,.0f}",
                        "Price_t": f"{Price_b_t:,.2f}",
                        "Price_t-1": f"{Price_b_tm1:,.2f}",
                        "Market_Effect": f"{Market_Effect:,.0f}",
                        "Share_Effect": f"{Share_Effect:,.0f}",
                        "Price_Effect": f"{Price_Effect:,.0f}",
                        "Residual": f"{Residual:,.0f}",
                    })
            else:
                st.info("ΔSales waterfall unavailable (need at least two periods).")
        except Exception:
            st.info("ΔSales waterfall unavailable for this selection.")
    # Linkage note for units
    st.caption("Units alignment: Potential/Actual/Headroom (u) in the waterfalls use the same values as the Funnel above.")

    with st.expander("Patients ↔ Units assumption"):
        st.markdown(
            f"""
            - Units shown in the funnel and Headroom waterfall are treated as patient-equivalents.
            - Conversion factor (resolved for current selection) = `{units_per_patient:.2f}`.
            - To change it, edit `headroom/config/scoring.yaml` — supported keys (checked in order):
              1) `assumptions.units_per_patient_equiv_by.market_brand["{sel_market}::{sel_brand}"]`
              2) `assumptions.units_per_patient_equiv_by.brand["{sel_brand}"]`
              3) `assumptions.units_per_patient_equiv_by.market["{sel_market}"]`
              4) `assumptions.units_per_patient_equiv` (global default)
            - Example:
              ```yaml
              assumptions:
                units_per_patient_equiv: 1.0  # global fallback
                units_per_patient_equiv_by:
                  market_brand:
                    "DE::{sel_brand}": 1.20
                  brand:
                    "{sel_brand}": 0.90
                  market:
                    "{sel_market}": 1.10
                  by_market:
                    DE:
                      {sel_brand}: 1.15
              ```
            - You can later make this brand-specific.
            """
        )

st.subheader("Market Metrics (Ranked)")
df_display = view if "DQ_Missing" not in view.columns else view[~view["DQ_Missing"]]
# Sort by Priority_Score (already renamed)
df_display_sorted = df_display.sort_values("Priority_Score", ascending=False)
# Add Reach to the table for quick scan if available
if "Reach_s" in df_display_sorted.columns:
    try:
        df_display_sorted["Reach (0–1)"] = df_display_sorted["Reach_s"].astype(float).round(2)
    except Exception:
        # fallback: leave as-is if conversion fails
        df_display_sorted["Reach (0–1)"] = df_display_sorted["Reach_s"]
st.dataframe(df_display_sorted, use_container_width=True)
with st.expander("How to read this table (Ceiling vs Access vs Reach)"):
    st.markdown(
        """
        **What you’re seeing**  
        A ranked view of markets/brands by **Priority Score (0–1)**.  
        Higher score = bigger prize, easier access, and better coverage.  

        **How the story builds**  
        1. 🔺 **Potential (size):** Adoption Ceiling caps eligible patients →  
           `Potential = Eligible × Adoption_Ceiling`.  
        2. 🟩 **Headroom (prize):** Untapped gap between Potential and Actual treated → shown in **units** and **value**.  
        3. 🔓📣 **Ease & coverage:**  
           - 🔓 **Access_Score** = payer/reimbursement ease.  
           - 📣 **Reach** = execution coverage (`customers_reached / customers_targeted` → scaled to `Reach_s`).  
           In v1, Access and Reach do *not* cap potential; they guide prioritization.  
        4. ⭐ **Priority (blend):**  
           `Priority = 0.5·HR_s + 0.3·(Acc_s·(1+Growth)) + 0.2·Reach_s` (defaults).  
           All components normalized 0–1 within the current dataset.  

        ```text
        Potential
          └─ Headroom (Potential − Actual)
              └─ Access & Reach (ease & coverage)
                  └─ Priority (blend)
        ```

        **How to use it**  
        Sort by **Priority**, then scan **Access** and **Reach** to see:  
        - 🔓 Is the gap blocked by **payer constraints**?  
        - 📣 Is it limited by **execution/coverage**?  
        - 🔺/🟩 Or is it simply a **large but already well-served** market?  

        **Learn more**  
        See [business_logic.md](docs/business_logic.md) and [data_usage.md](docs/data_usage.md).
        """
    )

row = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand) & (metrics["quarter"] == sel_quarter)]
if row.empty:
    st.info("No data for selection.")
else:
    r = row.iloc[0]
    # KPI cards (formatting for currency)
    fmt = scoring_cfg.get("formatting", {})
    currency = fmt.get("headroom_currency", "EUR")

    # KPI badges (Sales, Headroom), and the Access, Reach, Priority gauges
    sr_access = float(r.get("access_score", 0.0))
    sr_reach = float(r.get("Reach_s", 0.0))
    sr_roi = float(r.get("Priority_Score", 0.0))
    sr_sales_value = float(r.get("units_sold", 0.0)) * float(r.get("price_per_unit", 0.0))
    sr_headroom_value = float(r.get("Headroom_Value", 0.0))

    thr_cfg = scoring_cfg.get("thresholds", {})
    thr_acc = float(thr_cfg.get("accelerate", 0.70))
    thr_nur = float(thr_cfg.get("nurture", 0.50))

    # KPI badges were moved next to the Funnel at the top to start the story
    g1, g2, g3 = st.columns(3)
    with g1:
        gfig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sr_access,
            number={"valueformat": ".2f"},
            title={"text": ""},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#4C78A8"},
                "steps": [
                    {"range": [0, max(0.0, min(thr_nur, 1.0))], "color": "#FDE2E1"},
                    {"range": [max(0.0, min(thr_nur, 1.0)), max(0.0, min(thr_acc, 1.0))], "color": "#FFE9C6"},
                    {"range": [max(0.0, min(thr_acc, 1.0)), 1], "color": "#DDF2E0"},
                ],
            },
        ))
        gfig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=0))
        st.markdown("**Access (0–1)**")
        st.plotly_chart(gfig, use_container_width=True)
        with st.expander("What is Access Score?"):
            st.markdown(
                """
                - Access Score is a 0–1 proxy for how easy it is to capture headroom given payer/reimbursement conditions.
                - Higher = easier to win patients (e.g., fewer restrictions, better coverage, simpler prior auth).
                - We use it as the "ease-of-capture" component inside the Priority Score.
                - Contrast with Adoption Ceiling: Adoption Ceiling caps Potential_Units (volume). Access does not change Potential_Units in v1; it affects Priority (ordering). An explicit `accessible_fraction` could cap potential in v2.
                - See docs: `headroom/docs/glossary.md`, `headroom/docs/data_usage.md`.
                """
            )
    with g2:
        rchfig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sr_reach,
            number={"valueformat": ".2f"},
            title={"text": ""},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#72B7B2"},
                "steps": [
                    {"range": [0, max(0.0, min(thr_nur, 1.0))], "color": "#FDE2E1"},
                    {"range": [max(0.0, min(thr_nur, 1.0)), max(0.0, min(thr_acc, 1.0))], "color": "#FFE9C6"},
                    {"range": [max(0.0, min(thr_acc, 1.0)), 1], "color": "#DDF2E0"},
                ],
            },
        ))
        rchfig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=0))
        st.markdown("**Reach (0–1)**")
        st.plotly_chart(rchfig, use_container_width=True)
        with st.expander("How is Reach computed?"):
            st.markdown(
                """
                - Reach measures the fraction of targeted customers reached in the period.
                - Raw reach = customers_reached / customers_targeted (safe-handled when targeted is 0).
                - We normalize to 0–1 across the dataset (Reach_s) for comparability.
                - Reach complements Access by capturing execution/coverage.
                """
            )
    with g3:
        rfig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sr_roi,
            number={"valueformat": ".2f"},
            title={"text": ""},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#54A24B"},
                "steps": [
                    {"range": [0, max(0.0, min(thr_nur, 1.0))], "color": "#FDE2E1"},
                    {"range": [max(0.0, min(thr_nur, 1.0)), max(0.0, min(thr_acc, 1.0))], "color": "#FFE9C6"},
                    {"range": [max(0.0, min(thr_acc, 1.0)), 1], "color": "#DDF2E0"},
                ],
            },
        ))
        rfig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=0))
        st.markdown("**Priority Score (0–1)**")
        st.plotly_chart(rfig, use_container_width=True)
        with st.expander("How do we compute Priority Score?"):
            st.markdown(
                """
                Plain English:
                - We combine three ideas:
                  - Size of prize: bigger Headroom Value → higher score.
                  - Ease of capture: higher Access Score → higher score.
                  - Execution/coverage: higher Reach → higher score.
                - If the market is growing, we give a modest boost to the ease-of-capture part.
                - We normalize numbers to 0–1 so scores are comparable within the current dataset.
                - We then blend them using configurable weights.

                Compact formula (for reference):
                - Priority = w_hr × HR_s + w_sig × (Sig_s × (1 + growth_effect)) + w_reach × Reach_s
                  - HR_s = scaled Headroom_Value (0–1)
                  - Sig_s = scaled Access Score (0–1)
                  - Reach_s = scaled Reach (customers_reached/customers_targeted) (0–1)
                  - growth_effect = clipped growth_trend (negatives floored)
                - Defaults: w_hr=0.5, w_sig=0.3, w_reach=0.2. Configure weights and growth floor in `headroom/config/scoring.yaml`.
                - Details: `headroom/docs/business_logic.md` → Priority Score.
                """
            )


            Share_tm1 = (Units_b_tm1 / Units_m_tm1) if Units_m_tm1 > 0 else 0.0
            Share_t = (Units_b_t / Units_m_t) if Units_m_t > 0 else 0.0

            MarketEff = (Units_m_t - Units_m_tm1) * Share_tm1 * Price_b_tm1
            ShareEff = (Share_t - Share_tm1) * Units_m_t * Price_b_tm1
            PriceEff = (Price_b_t - Price_b_tm1) * Units_b_t
            DeltaSales = Sales_b_t - Sales_b_tm1
            Residual = DeltaSales - (MarketEff + ShareEff + PriceEff)

            hovertexts = [
                f"Market Growth: ΔMarketUnits={Units_m_t-Units_m_tm1:,.0f} · Share(t-1)={Share_tm1:.2%} · Price(t-1)={Price_b_tm1:,.2f}",
                f"Patient Share: ΔShare={Share_t-Share_tm1:.2%} · MarketUnits(t)={Units_m_t:,.0f} · Price(t-1)={Price_b_tm1:,.2f}",
                f"Price: ΔPrice={Price_b_t-Price_b_tm1:,.2f} · Units(t)={Units_b_t:,.0f}",
                f"Residual: ΔSales − (Market+Share+Price)",
                f"ΔSales: Sales(t)−Sales(t-1)={Sales_b_t:,.0f}−{Sales_b_tm1:,.0f}",
            ]
            wf = go.Figure(go.Waterfall(
                name="ΔSales",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=["Market Growth", "Patient Share", "Price", "Other/Residual", "ΔSales"],
                textposition="outside",
                text=[f"{MarketEff:,.0f}", f"{ShareEff:,.0f}", f"{PriceEff:,.0f}", f"{Residual:,.0f}", f"{DeltaSales:,.0f}"],
                y=[MarketEff, ShareEff, PriceEff, Residual, DeltaSales],
                connector={"line": {"color": "#A0A0A0"}},
                hovertext=hovertexts,
                hoverinfo="text+name",
            ))
    # Below: Trend grid (2x2) for Sales, Headroom, Price, Patient Share
    st.subheader("Trends – Sales, Headroom, Price, Patient Share")
    trend = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand)].copy()
    if not trend.empty:
        try:
            trend = trend.sort_values("quarter")
        except Exception:
            pass
        trend["Sales_Value"] = trend["units_sold"].astype(float) * trend["price_per_unit"].astype(float)
        if "net_sales" in trend.columns:
            with pd.option_context('mode.use_inf_as_na', True):
                trend["Net_Price_per_Unit"] = (
                    (trend["net_sales"].astype(float) / trend["units_sold"].replace(0, pd.NA).astype(float))
                )
            trend["Price_for_Charts"] = trend["Net_Price_per_Unit"].fillna(trend["price_per_unit"].astype(float))
        else:
            trend["Price_for_Charts"] = trend["price_per_unit"].astype(float)

        # Precompute quarters caption
        try:
            quarters_sorted = sorted(trend["quarter"].unique().tolist(), key=lambda q: (int(str(q)[:4]), int(str(q)[-1:])))
            trend_caption = f"Trend period: {quarters_sorted[0]} → {quarters_sorted[-1]}"
        except Exception:
            quarters_sorted = list(trend["quarter"].unique())
            trend_caption = ""

        # Row 1
        t11, t12 = st.columns(2)
        with t11:
            sfig = go.Figure()
            sfig.add_trace(go.Scatter(
                x=trend["quarter"], y=trend["Sales_Value"], mode="lines+markers",
                name="Sales Value (€)", line=dict(color="#54A24B"), marker=dict(size=6)
            ))
            sfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quarter", yaxis_title="Sales Value", yaxis=dict(rangemode="tozero"))
            st.plotly_chart(sfig, use_container_width=True)
            if trend_caption:
                st.caption(trend_caption)
        with t12:
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(
                x=trend["quarter"], y=trend["Headroom_Value"], mode="lines+markers",
                name="Headroom (€)", line=dict(color="#4C78A8"), marker=dict(size=6)
            ))
            hfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quarter", yaxis_title="Headroom Value", yaxis=dict(rangemode="tozero"))
            st.plotly_chart(hfig, use_container_width=True)
            if trend_caption:
                st.caption(trend_caption)

        # Row 2
        t21, t22 = st.columns(2)
        with t21:
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(
                x=trend["quarter"], y=trend["Price_for_Charts"], mode="lines+markers",
                name="Price per Unit", line=dict(color="#E45756"), marker=dict(size=6)
            ))
            pfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quarter", yaxis_title="Price per Unit", yaxis=dict(rangemode="tozero"))
            st.plotly_chart(pfig, use_container_width=True)
            if trend_caption:
                price_basis = "Net price (net_sales/units)" if "net_sales" in trend.columns else "List price"
                st.caption(f"{trend_caption} · {price_basis}")
        with t22:
            try:
                market_totals = (
                    metrics[metrics["market"].eq(sel_market)]
                    .groupby("quarter", as_index=False)["units_sold"].sum()
                    .rename(columns={"units_sold": "Market_Units"})
                )
                trend_share = trend.merge(market_totals, on="quarter", how="left")
                trend_share["Patient_Share_pct"] = (
                    (trend_share["units_sold"].astype(float) / trend_share["Market_Units"].replace(0, pd.NA).astype(float)) * 100.0
                )
                shfig = go.Figure()
                shfig.add_trace(go.Scatter(
                    x=trend_share["quarter"], y=trend_share["Patient_Share_pct"], mode="lines+markers",
                    name="Patient Share (%)", line=dict(color="#72B7B2"), marker=dict(size=6)
                ))
                shfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quarter", yaxis_title="Patient Share (%)", yaxis=dict(rangemode="tozero"))
                st.plotly_chart(shfig, use_container_width=True)
                if trend_caption:
                    st.caption(f"{trend_caption} · Share = Units_brand / Market_Units")
            except Exception:
                st.info("Could not compute Patient Share trend (need at least two quarters or valid market totals).")
    else:
        st.info("No trend data available for the selected market/brand.")

    # Recommendations below in a framed box
    st.subheader("Insights")
    st.caption("Descriptive, data-only insights (no prescriptions).")
    # Load optional LLM config for defaults
    try:
        llm_cfg_ins = load_yaml(os.path.join(BASE_DIR, "config", "llm.yaml"))
    except Exception:
        llm_cfg_ins = {}
    prov_ins = (llm_cfg_ins.get("provider") or os.getenv("LLM_PROVIDER") or "openai")
    model_ins = (llm_cfg_ins.get("model") or os.getenv("LLM_MODEL") or "gpt-4o-mini")
    gen_insights = st.button("Generate insights")

    if gen_insights:
        try:
            quarters_list = trend["quarter"].astype(str).tolist() if "trend" in locals() else []
            sales_vals = trend.get("Sales_Value", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            headroom_vals = trend.get("Headroom_Value", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            price_vals = trend.get("Price_for_Charts", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            # Keep last 8
            quarters_last = quarters_list[-8:]
            sales_last = sales_vals[-8:]
            headroom_last = headroom_vals[-8:]
            price_last = price_vals[-8:]
            trends_ctx = {
                "quarters": quarters_last,
                "sales_value": sales_last,
                "headroom_value": headroom_last,
                "price": price_last,
                "sales_value_map": {q: v for q, v in zip(quarters_last, sales_last)},
                "headroom_value_map": {q: v for q, v in zip(quarters_last, headroom_last)},
                "price_map": {q: v for q, v in zip(quarters_last, price_last)},
            }

            context_ins = {
                "mode": "insights",
                "params": {"temperature": 0.0, "max_tokens": 900},
                "selection": {"market": sel_market, "brand": sel_brand, "quarter": sel_quarter},
                "kpis": {
                    "units_sold": float(r.get("units_sold", 0.0)),
                    "price_per_unit": float(r.get("price_per_unit", 0.0)),
                    "Headroom_Value": float(r.get("Headroom_Value", 0.0)),
                    "access_score": float(r.get("access_score", 0.0)),
                    "ROI_Score": float(r.get("ROI_Score", 0.0)),
                },
                "trends": trends_ctx,
                "decomp": locals().get("DeltaSales") and {
                    "market_eff": float(locals().get("MarketEff", 0.0)),
                    "share_eff": float(locals().get("ShareEff", 0.0)),
                    "price_eff": float(locals().get("PriceEff", 0.0)),
                    "delta_sales": float(locals().get("DeltaSales", 0.0)),
                } or None,
                "thresholds": scoring_cfg.get("thresholds", {}),
                "currency": scoring_cfg.get("formatting", {}).get("headroom_currency", "EUR"),
                "provider": prov_ins,
                "model": model_ins,
            }

            with st.spinner("Generating insights…"):
                context_key = json.dumps(context_ins, ensure_ascii=False, sort_keys=True)
                insight_res = _cached_llm_call(context_key)

            if isinstance(insight_res, dict):
                if insight_res.get("summary"):
                    st.markdown(f"**Summary**: {insight_res['summary']}")
                bullets = insight_res.get("bullets") or []
                if bullets:
                    st.markdown("**Key insights**")
                    for b in bullets:
                        text = b.get("text") if isinstance(b, dict) else str(b)
                        cites = b.get("data_citations", []) if isinstance(b, dict) else []
                        cite_str = f" _(refs: {', '.join(map(str, cites))})_" if cites else ""
                        st.markdown(f"- {text}{cite_str}")
                else:
                    # Fallback to legacy 'options' observations
                    obs = insight_res.get("options", []) or []
                    if obs:
                        st.markdown("**Observations**")
                        for o in obs:
                            st.markdown(f"- {o.get('title') or o.get('rationale') or str(o)}")
            else:
                st.json(insight_res)
        except Exception as e:
            st.warning(f"Insights unavailable: {e}")

    # Optional: AI-generated insights & recommendations (LLM)
    st.divider()
    st.subheader("Brainstorm recommendations")
    # Load optional LLM config
    try:
        llm_cfg = load_yaml(os.path.join(BASE_DIR, "config", "llm.yaml"))
    except Exception:
        llm_cfg = {}
    prov_default = (llm_cfg.get("provider") or os.getenv("LLM_PROVIDER") or "openai")
    model_default = (llm_cfg.get("model") or os.getenv("LLM_MODEL") or "gpt-4o-mini")
    colp, colm = st.columns([1, 2])
    with colp:
        provider = st.selectbox("Provider", options=["openai"], index=["openai"].index(prov_default) if prov_default in ["openai"] else 0)
    with colm:
        base_models = ["gpt-4o", "gpt-4o-mini"]
        # Ensure default is visible
        if model_default not in base_models:
            base_models = [model_default] + [m for m in base_models if m != model_default]
        model_options = base_models
        default_idx = model_options.index(model_default) if model_default in model_options else 0
        selected_model = st.selectbox("Model", options=model_options, index=default_idx)
    user_note = st.text_area("Focus (optional)", placeholder="e.g., prioritize access barriers vs. price actions")
    gen = st.button("Brainstorm recommendations")

    if gen:
        # Build context for LLM
        try:
            # Trends may be empty; slice safely and include quarter-keyed maps
            quarters_list = trend["quarter"].astype(str).tolist() if "trend" in locals() else []
            sales_vals = trend.get("Sales_Value", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            headroom_vals = trend.get("Headroom_Value", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            price_vals = trend.get("Price_for_Charts", pd.Series(dtype=float)).astype(float).tolist() if "trend" in locals() else []
            quarters_last = quarters_list[-8:]
            sales_last = sales_vals[-8:]
            headroom_last = headroom_vals[-8:]
            price_last = price_vals[-8:]
            trends_ctx = {
                "quarters": quarters_last,
                "sales_value": sales_last,
                "headroom_value": headroom_last,
                "price": price_last,
                "sales_value_map": {q: v for q, v in zip(quarters_last, sales_last)},
                "headroom_value_map": {q: v for q, v in zip(quarters_last, headroom_last)},
                "price_map": {q: v for q, v in zip(quarters_last, price_last)},
            }

            context = {
                "selection": {"market": sel_market, "brand": sel_brand, "quarter": sel_quarter},
                "kpis": {
                    "units_sold": float(r.get("units_sold", 0.0)),
                    "price_per_unit": float(r.get("price_per_unit", 0.0)),
                    "Headroom_Value": float(r.get("Headroom_Value", 0.0)),
                    "access_score": float(r.get("access_score", 0.0)),
                    "ROI_Score": float(r.get("ROI_Score", 0.0)),
                },
                "trends": trends_ctx,
                "decomp": locals().get("DeltaSales") and {
                    "market_eff": float(locals().get("MarketEff", 0.0)),
                    "share_eff": float(locals().get("ShareEff", 0.0)),
                    "price_eff": float(locals().get("PriceEff", 0.0)),
                    "delta_sales": float(locals().get("DeltaSales", 0.0)),
                } or None,
                "thresholds": scoring_cfg.get("thresholds", {}),
                "currency": scoring_cfg.get("formatting", {}).get("headroom_currency", "EUR"),
                "notes": user_note,
                "provider": provider,
                "model": selected_model,
                "mode": "recommendations",
                "params": {"temperature": float(llm_cfg.get("params", {}).get("temperature", 0.2)) if 'llm_cfg' in locals() else 0.2, "max_tokens": int(llm_cfg.get("params", {}).get("max_tokens", 900)) if 'llm_cfg' in locals() else 900},
            }

            # Visual cue when focus is provided
            if (user_note or "").strip():
                st.info("Focus applied: " + (user_note or "").strip())

            with st.spinner("Brainstorming recommendations…"):
                context_key = json.dumps(context, ensure_ascii=False, sort_keys=True)
                ai_res = _cached_llm_call(context_key)

            # Render AI output
            if isinstance(ai_res, dict):
                if ai_res.get("summary"):
                    st.markdown(f"**Summary**: {ai_res['summary']}")
                opts = ai_res.get("options", []) or []
                for i, opt in enumerate(opts, start=1):
                    st.markdown(f"### Option {i}: {opt.get('title', 'Untitled')}")
                    if opt.get("rationale"):
                        st.markdown(opt["rationale"]) 
                    eff = opt.get("expected_effect") or {}
                    if eff:
                        st.markdown(f"- Expected effect: sales Δ {eff.get('sales_delta', 'n/a')}, headroom Δ {eff.get('headroom_delta', 'n/a')}")
                    # Focus alignment if present
                    if opt.get("focus_alignment") is not None:
                        try:
                            fa = float(opt.get("focus_alignment"))
                            st.markdown(f"- Focus alignment: {fa:.2f}")
                        except Exception:
                            st.markdown(f"- Focus alignment: {opt.get('focus_alignment')}")
                    if opt.get("why_aligned"):
                        st.markdown(f"- Why aligned: {opt.get('why_aligned')}")
                    risks = opt.get("risks") or []
                    if risks:
                        st.markdown("- Risks:")
                        for rk in risks:
                            st.markdown(f"  - {rk}")
                    cites = opt.get("data_citations") or []
                    if cites:
                        st.caption("Data citations: " + ", ".join(map(str, cites)))
                if ai_res.get("confidence") is not None:
                    st.caption(f"Model: {provider}/{selected_model} · Confidence: {ai_res['confidence']}")
            else:
                st.json(ai_res)

        except Exception as e:
            st.warning(f"AI recommendations unavailable: {e}")


