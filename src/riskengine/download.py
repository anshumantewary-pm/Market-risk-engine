"""Download daily adjusted closes from Stooq's CSV endpoint.

Usage::

    python -m riskengine.download [--config config/portfolio.yaml] [--out data/raw/prices.csv]

Parsing is kept separate from fetching so it can be tested on a fixture file without
network access.
"""

from __future__ import annotations

import io

import pandas as pd

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


class DownloadError(RuntimeError):
    """Raised when a ticker cannot be downloaded or its CSV cannot be parsed."""


def stooq_symbol(ticker: str) -> str:
    """Stooq symbol for a US-listed ticker, e.g. ``SPY`` -> ``spy.us``."""
    return f"{ticker.strip().lower()}.us"


def parse_stooq_csv(text: str, ticker: str) -> pd.Series:
    """Parse one Stooq daily CSV into a Series of closes indexed by date.

    Stooq returns columns ``Date,Open,High,Low,Close,Volume``. An unknown symbol returns a
    short text body such as ``No data`` instead of a CSV.
    """
    body = text.strip()
    if not body or body.lower().startswith("no data") or "Date" not in body.splitlines()[0]:
        raise DownloadError(f"no data returned for {ticker}: {body[:60]!r}")
    df = pd.read_csv(io.StringIO(body))
    if "Close" not in df.columns:
        raise DownloadError(f"no Close column for {ticker}; got {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = ticker
    return s
