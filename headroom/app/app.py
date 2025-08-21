"""Headroom Pretotype demo app — v1 pilot context: Germany (DE), oncology brands."""
import os
import sys
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    quarters = sorted(
        data["markets"][(data["markets"]["market"] == sel_market) & (data["markets"]["brand"] == sel_brand)]["quarter"].unique()
    )
    sel_quarter = st.selectbox("Quarter", quarters)
    apply_filters = st.checkbox("Apply filters to views", value=True)
    st.divider()
    st.subheader("Help & Docs")
    st.markdown(
        """
        - [Business Logic](../docs/business_logic.md)
        - [Data Usage](../docs/data_usage.md)
        - [Glossary](../docs/glossary.md)
        """
    )

metrics = compute_metrics(data["markets"], data["actuals"], data["drivers"], scoring_cfg=scoring_cfg)

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

st.subheader("Market Metrics (Ranked)")
df_display = view if "DQ_Missing" not in view.columns else view[~view["DQ_Missing"]]
st.dataframe(df_display.sort_values("ROI_Score", ascending=False), use_container_width=True)

row = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand) & (metrics["quarter"] == sel_quarter)]
if row.empty:
    st.info("No data for selection.")
else:
    r = row.iloc[0]
    # KPI cards (formatting for currency)
    fmt = scoring_cfg.get("formatting", {})
    currency = fmt.get("headroom_currency", "EUR")

    # KPI badges (Sales, Headroom first), Access gauge to the right
    sr_access = float(r.get("access_score", 0.0))
    sr_sales_value = float(r.get("units_sold", 0.0)) * float(r.get("price_per_unit", 0.0))
    sr_headroom_value = float(r.get("Headroom_Value", 0.0))

    thr_cfg = scoring_cfg.get("thresholds", {})
    thr_acc = float(thr_cfg.get("accelerate", 0.70))
    thr_nur = float(thr_cfg.get("nurture", 0.50))

    b1, b2, b3 = st.columns([1, 1, 2])
    b1.metric(f"Sales Value ({currency})", f"{sr_sales_value:,.0f}")
    b2.metric(f"Headroom ({currency})", f"{sr_headroom_value:,.0f}")
    with b3:
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
    # Top row: Funnel (left) and Decomposition (right)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Funnel – Patients to Units")
        patients_total = float(r["patients_total"])
        patients_eligible = float(r["patients_eligible"])
        units_sold = float(r["units_sold"])

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

        fig = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textposition="inside",
            textinfo="value+percent initial",
            hoverinfo="text+name",
            textfont=dict(color="white"),
            hovertext=hover_text,
            marker={"color": ["#4C78A8", "#72B7B2", "#54A24B"]},
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Period: {sel_quarter}. Note: 'Accessible Patients' is modeled via Access_Score in v1 and not shown as a separate funnel stage.")

        with st.expander("Funnel definitions"):
            st.markdown(
                """
                - a) Total Patient Population — epidemiology base (incidence/prevalence)
                - b) Eligible Patients — meet label/guideline criteria for the brand
                - c) Accessible Patients — within payer coverage/reimbursement constraints (modeled via Access_Score in v1)
                - d) Treated Patients — on the selected brand’s therapy in the period (proxied by that brand’s units sold)
                - e) Sales (Units × Price) — commercial output from treated units at net price
                - f) Headroom = Potential – Actual — gap between potential treated and actual treated; size of prize (units and €)
                """
            )

    with c2:
        # Past Performance Decomposition (Waterfall)
        st.subheader("Past Performance Decomposition – ΔSales components")
        # Build current and previous quarter context for selected market/brand
        brand_hist = metrics[metrics["market"].eq(sel_market)].copy()
        if not brand_hist.empty:
            # Sort quarters reliably (expects format YYYYQx)
            try:
                brand_hist["_q_sort"] = brand_hist["quarter"].astype(str).apply(lambda q: (int(q[:4]), int(q[-1:])))
                brand_hist = brand_hist.sort_values(["brand", "_q_sort"])  # ensure per-brand order
            except Exception:
                brand_hist = brand_hist.sort_values(["brand", "quarter"])  # fallback

            # Pick latest available quarter for the selected brand
            bh = brand_hist[brand_hist["brand"].eq(sel_brand)]
            if len(bh) >= 2:
                # Determine t (latest) and t-1
                try:
                    q_list = sorted(bh["quarter"].unique().tolist(), key=lambda q: (int(str(q)[:4]), int(str(q)[-1:])))
                except Exception:
                    q_list = list(bh["quarter"].unique().tolist())
                q_t = q_list[-1]
                q_tm1 = q_list[-2]

                b_t = bh[bh["quarter"].eq(q_t)].iloc[0]
                b_tm1 = bh[bh["quarter"].eq(q_tm1)].iloc[0]

                Units_b_t = float(b_t["units_sold"])
                Units_b_tm1 = float(b_tm1["units_sold"])
                # Prefer net price if net_sales present
                def _price(row):
                    try:
                        if "net_sales" in row.index and pd.notna(row["net_sales"]) and row["units_sold"] not in (0, None):
                            return float(row["net_sales"]) / float(row["units_sold"]) if float(row["units_sold"]) != 0 else float(row.get("price_per_unit", 0.0))
                    except Exception:
                        pass
                    return float(row.get("price_per_unit", 0.0))

                Price_b_t = _price(b_t)
                Price_b_tm1 = _price(b_tm1)
                Sales_b_t = Units_b_t * Price_b_t
                Sales_b_tm1 = Units_b_tm1 * Price_b_tm1

                # Market totals per quarter (sum across brands in same market)
                mkt = metrics[metrics["market"].eq(sel_market)]
                Units_m_t = float(mkt[mkt["quarter"].eq(q_t)]["units_sold"].sum())
                Units_m_tm1 = float(mkt[mkt["quarter"].eq(q_tm1)]["units_sold"].sum())

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
                wf.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(rangemode="tozero"),
                )
                st.plotly_chart(wf, use_container_width=True)
                st.caption(
                    f"Decomposition period: {q_tm1} → {q_t} (brand: {sel_brand}, market: {sel_market}) · "
                    f"Share: {Share_tm1:.2%} → {Share_t:.2%} · Market Units: {Units_m_tm1:,.0f} → {Units_m_t:,.0f} · "
                    f"Price basis: {'net' if ('net_sales' in metrics.columns) else 'list'}"
                )

                with st.expander("Decomposition inputs"):
                    st.write({
                        "Units_brand": {str(q_tm1): Units_b_tm1, str(q_t): Units_b_t},
                        "Price": {str(q_tm1): Price_b_tm1, str(q_t): Price_b_t},
                        "Sales": {str(q_tm1): Sales_b_tm1, str(q_t): Sales_b_t},
                        "Market_Units": {str(q_tm1): Units_m_tm1, str(q_t): Units_m_t},
                        "Share": {str(q_tm1): Share_tm1, str(q_t): Share_t},
                    })
            else:
                st.info("Not enough history to decompose past performance (need at least two quarters).")
        else:
            st.info("No data available to compute decomposition.")

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
    st.subheader("Recommendations")
    recs = recommend_actions(view, thresholds=scoring_cfg.get("thresholds", {}))
    if len(recs) == 0:
        st.info("No specific recommendations yet — tune rules in src/rules.py")
    else:
        box_start = """
<div style="border:1px solid #e6e6e6; padding:12px; border-radius:8px; background:#fafafa;">
"""
        box_end = "</div>"
        items = "\n".join([f"- {rtxt}" for rtxt in recs])
        st.markdown(box_start + items + box_end, unsafe_allow_html=True)


