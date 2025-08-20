import os
import pandas as pd
from typing import Dict


def _resolve(path: str, base_dir: str | None = None) -> str:
    # Resolve relative to base_dir (e.g., headroom/) if provided
    if base_dir and not os.path.isabs(path):
        return os.path.normpath(os.path.join(base_dir, path))
    return os.path.normpath(path)


def load_public_data(ds_cfg: Dict[str, object], base_dir: str | None = None) -> Dict[str, pd.DataFrame]:
    """
    ds_cfg schema:
      public_data_dir: data/public
      files: {markets, actuals, market_drivers}
    """
    data_dir = ds_cfg.get("public_data_dir", "data/public")
    files = ds_cfg.get("files", {}) or {}

    def p(filename: str) -> str:
        return _resolve(os.path.join(data_dir, filename), base_dir)

    markets = pd.read_csv(p(files.get("markets", "markets.csv")))
    actuals = pd.read_csv(p(files.get("actuals", "actuals.csv")))
    drivers = pd.read_csv(p(files.get("market_drivers", "market_drivers.csv")))

    return {"markets": markets, "actuals": actuals, "drivers": drivers}
