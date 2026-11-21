"""Read, write and validate the prices file.

File contract (``data/raw/prices.csv``): a ``Date`` column with ISO dates, then one column
of adjusted closing prices per ticker. The real downloader and the synthetic generator both
write exactly this format, so everything downstream is source-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PRICES_PATH = Path("data/raw/prices.csv")


class DataError(ValueError):
    """Raised when a prices file breaks the file contract."""


def read_prices_csv(path: str | Path = DEFAULT_PRICES_PATH) -> pd.DataFrame:
    """Read a prices CSV into a DataFrame indexed by date (no validation)."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"prices file not found: {path}. Run `python -m riskengine.download` for real "
            "data or `python -m riskengine.pipeline --sample` for synthetic data."
        )
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise DataError(f"{path} has no 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def save_prices_csv(prices: pd.DataFrame, path: str | Path = DEFAULT_PRICES_PATH) -> Path:
    """Write prices in the file contract format and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")
    return path
