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
    # Funnel and Quarter Trend side-by-side
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
        st.caption("Note: 'Accessible Patients' is modeled via Access_Score in v1 and not shown as a separate funnel stage.")

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
        st.subheader("Trend – Sales and Headroom by Quarter")
        trend = metrics[(metrics["market"] == sel_market) & (metrics["brand"] == sel_brand)].copy()
        if not trend.empty:
            # Ensure a stable quarter order; if already sorted, this is a no-op
            try:
                trend = trend.sort_values("quarter")
            except Exception:
                pass
            # Compute Sales (Total) Value
            trend["Sales_Value"] = trend["units_sold"].astype(float) * trend["price_per_unit"].astype(float)

            # Chart 1: Sales (Total) Value
            sfig = go.Figure()
            sfig.add_trace(go.Scatter(
                x=trend["quarter"], y=trend["Sales_Value"], mode="lines+markers",
                name="Sales Value (€)", line=dict(color="#54A24B"), marker=dict(size=6)
            ))
            sfig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Quarter",
                yaxis_title="Sales Value",
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(sfig, use_container_width=True)

            # Chart 2: Headroom Value
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(
                x=trend["quarter"], y=trend["Headroom_Value"], mode="lines+markers",
                name="Headroom (€)", line=dict(color="#4C78A8"), marker=dict(size=6)
            ))
            hfig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Quarter",
                yaxis_title="Headroom Value",
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(hfig, use_container_width=True)
        else:
            st.info("No trend data for this Market/Brand.")

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


