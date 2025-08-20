from typing import Dict
import numpy as np
import pandas as pd


def _safe_minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    xmin, xmax = float(x.min()), float(x.max())
    if np.isclose(xmin, xmax):
        return pd.Series([0.5] * len(x), index=x.index)
    return (x - xmin) / (xmax - xmin)


def compute_metrics(df_markets: pd.DataFrame, df_actuals: pd.DataFrame, df_drivers: pd.DataFrame, scoring_cfg: Dict) -> pd.DataFrame:
    """
    Compute Potential, Headroom, and ROI scoring for market/brand/quarter rows.

    Inputs are joined on ['market','brand','quarter'].
    """
    keys = ["market", "brand", "quarter"]
    df = (
        df_markets.merge(df_actuals, on=keys, how="left")
                  .merge(df_drivers, on=keys, how="left")
    )

    # Data quality: flag rows with missing critical inputs BEFORE filling defaults
    critical_cols = [
        "patients_eligible",
        "adoption_ceiling",
        "price_per_unit",
        "units_sold",
        "access_score",
    ]
    # Ensure columns exist for the check
    for c in critical_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["DQ_Missing"] = df[critical_cols].isna().any(axis=1)

    # Optional price cap for outliers (per finance guidance)
    price_cap = None
    try:
        price_cap = float((scoring_cfg or {}).get("price_cap"))
    except (TypeError, ValueError):
        price_cap = None

    # Fill defaults
    df["patients_total"] = df["patients_total"].fillna(0)
    df["patients_eligible"] = df["patients_eligible"].fillna(0)
    df["adoption_ceiling"] = df["adoption_ceiling"].fillna(0.0)
    df["price_per_unit"] = df["price_per_unit"].fillna(0.0)
    if price_cap is not None and np.isfinite(price_cap):
        df["price_per_unit"] = df["price_per_unit"].clip(upper=price_cap)
    df["units_sold"] = df["units_sold"].fillna(0)
    df["access_score"] = df["access_score"].fillna(0.0)
    df["growth_trend"] = df["growth_trend"].fillna(0.0)

    # Core formulas
    df["Potential_Units"] = df["patients_eligible"] * df["adoption_ceiling"]
    df["Headroom_Units"] = (df["Potential_Units"] - df["units_sold"]).clip(lower=0)
    df["Headroom_Value"] = df["Headroom_Units"] * df["price_per_unit"]
    # Flags per documentation
    df["Flag_Actual_GT_Potential"] = df["units_sold"] > df["Potential_Units"]

    # Scoring
    weights = (scoring_cfg or {}).get("weights", {})
    w_hr = float(weights.get("headroom_value", 0.6))
    w_sig = float(weights.get("access_signal", 0.4))
    growth_clip_min = float((scoring_cfg or {}).get("growth_clip_min", 0.0))

    df["HR_s"] = _safe_minmax(df["Headroom_Value"])  # 0..1
    df["Sig_s"] = _safe_minmax(df["access_score"])   # 0..1
    df["growth_effect"] = df["growth_trend"].clip(lower=growth_clip_min)
    df["ROI_Score"] = w_hr * df["HR_s"] + w_sig * (df["Sig_s"] * (1 + df["growth_effect"]))
    # Acceptance: ROI must be in [0,1]
    df["ROI_Score"] = df["ROI_Score"].clip(lower=0.0, upper=1.0)

    # Output ordering
    cols = [
        "market", "brand", "quarter",
        "patients_total", "patients_eligible", "adoption_ceiling", "price_per_unit",
        "units_sold", "Potential_Units", "Headroom_Units", "Headroom_Value",
        "access_score", "growth_trend", "ROI_Score",
        "DQ_Missing", "Flag_Actual_GT_Potential",
    ]
    return df[cols]
