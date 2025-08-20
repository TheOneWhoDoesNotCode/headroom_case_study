import pandas as pd
from typing import List, Dict


def recommend_actions(metrics: pd.DataFrame, thresholds: Dict[str, float] | None = None) -> List[str]:
    """
    Recommend actions per row using ROI and headroom thresholds.

    thresholds schema (from scoring.yaml):
      accelerate: e.g., 0.70
      nurture: e.g., 0.50
    """
    thresholds = thresholds or {}
    thr_acc = float(thresholds.get("accelerate", 0.7))
    thr_nur = float(thresholds.get("nurture", 0.5))

    df = metrics.copy()
    # Exclude rows with data-quality issues if the flag exists
    if "DQ_Missing" in df.columns:
        df = df[~df["DQ_Missing"]]
    # Only consider opportunities with positive headroom value
    df = df[df["Headroom_Value"] > 0]
    df = df.sort_values("ROI_Score", ascending=False)

    recs: List[str] = []
    for _, row in df.iterrows():
        roi = float(row["ROI_Score"]) if "ROI_Score" in row else 0.0
        if roi >= thr_acc:
            label = "Accelerate"
        elif roi >= thr_nur:
            label = "Nurture"
        else:
            label = "Monitor"

        recs.append(
            f"{label}: {row['market']} · {row['brand']} · {row['quarter']} — Headroom €{row['Headroom_Value']:,.0f}, ROI {roi:.2f}"
        )

    # Return top 10 to keep the list readable
    return recs[:10]
