"""Read, write and validate the prices file.

File contract (``data/raw/prices.csv``): a ``Date`` column with ISO dates, then one column
of adjusted closing prices per ticker. The real downloader and the synthetic generator both
write exactly this format, so everything downstream is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class LoadReport:
    """What validation did to the raw file, so nothing happens silently."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    tickers: tuple[str, ...]
    trimmed_rows: int  # rows outside the common date range (a ticker had not started yet)
    dropped_rows: int  # rows inside the range where at least one ticker had no price


def validate_prices(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    max_gap_days: int = 10,
) -> tuple[pd.DataFrame, LoadReport]:
    """Check a prices frame against the file contract and return a clean copy.

    * the index must be strictly increasing, unique dates;
    * every requested ticker must be present, numeric and strictly positive;
    * the frame is restricted to the common date range where every ticker has a price;
    * missing prices inside that range are dropped row-wise (never forward-filled) and
      counted in the report;
    * a calendar gap longer than ``max_gap_days`` raises, because it usually means a
      broken download rather than a market holiday.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataError("prices must be indexed by dates")
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()].unique()[:3]
        raise DataError(f"duplicate dates in prices, e.g. {list(dupes.date)}")
    if not prices.index.is_monotonic_increasing:
        raise DataError("dates in prices must be sorted in increasing order")

    tickers = list(prices.columns) if tickers is None else list(tickers)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataError(f"tickers missing from prices: {missing}")
    df = prices[tickers].apply(pd.to_numeric, errors="coerce")
    bad_text = df.isna() & prices[tickers].notna()
    if bad_text.any().any():
        cols = list(df.columns[bad_text.any()])
        raise DataError(f"non-numeric prices in: {cols}")
    if (df <= 0).any().any():
        cols = list(df.columns[(df <= 0).any()])
        raise DataError(f"non-positive prices in: {cols}")

    starts = df.apply(pd.Series.first_valid_index)
    ends = df.apply(pd.Series.last_valid_index)
    if starts.isna().any():
        raise DataError(f"no prices at all for: {list(starts[starts.isna()].index)}")
    start, end = starts.max(), ends.min()
    if start >= end:
        raise DataError("tickers have no overlapping date range")
    in_range = df.loc[start:end]
    trimmed = len(df) - len(in_range)
    clean = in_range.dropna(how="any")
    dropped = len(in_range) - len(clean)

    if len(clean) < 2:
        raise DataError("fewer than two complete rows of prices")
    gaps = clean.index.to_series().diff().dt.days
    if gaps.max() > max_gap_days:
        at = gaps.idxmax()
        raise DataError(
            f"gap of {int(gaps.max())} calendar days in prices ending {at.date()} "
            f"(limit {max_gap_days})"
        )

    report = LoadReport(
        start=clean.index[0],
        end=clean.index[-1],
        n_rows=len(clean),
        tickers=tuple(tickers),
        trimmed_rows=trimmed,
        dropped_rows=dropped,
    )
    return clean.astype(float), report


def load_prices(
    path: str | Path = DEFAULT_PRICES_PATH,
    tickers: list[str] | None = None,
    max_gap_days: int = 10,
) -> tuple[pd.DataFrame, LoadReport]:
    """Read and validate a prices file in one step."""
    return validate_prices(read_prices_csv(path), tickers, max_gap_days)
