import pandas as pd
from src.compute import compute_metrics


def test_compute_metrics_basic():
    markets = pd.DataFrame({"market_id": [1, 2], "market": ["A", "B"]})
    actuals = pd.DataFrame({"market_id": [1, 2], "actual": [100, 0]})
    signals = pd.DataFrame({"market_id": [1, 2], "potential": [150, 50], "signal": [0.5, 1.0]})

    df = compute_metrics(markets, actuals, signals)

    # Headroom
    assert float(df.loc[df.market == "A", "headroom"].iloc[0]) == 50
    assert float(df.loc[df.market == "B", "headroom"].iloc[0]) == 50

    # Score defined and finite
    assert df["roi_score"].notna().all()
