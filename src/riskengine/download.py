"""Download daily adjusted closes from Stooq's CSV endpoint.

Usage::

    python -m riskengine.download [--config config/portfolio.yaml] [--out data/raw/prices.csv]

Parsing is kept separate from fetching so it can be tested on a fixture file without
network access.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from collections.abc import Callable

import pandas as pd

from riskengine.config import DEFAULT_CONFIG_PATH, load_config
from riskengine.data.load import DEFAULT_PRICES_PATH, save_prices_csv

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


def fetch_text(url: str, timeout: float = 30.0) -> str:
    """Fetch a URL and return its body as text (standard library only)."""
    req = urllib.request.Request(url, headers={"User-Agent": "market-risk-engine/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except OSError as exc:  # URLError, HTTPError and timeouts are all OSErrors
        raise DownloadError(f"could not fetch {url}: {exc}") from exc


def download_prices(
    tickers: list[str],
    fetch: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Download closes for every ticker and merge them on date (outer join).

    The loader later restricts the result to the common date range, so tickers with
    shorter histories are kept as-is here.
    """
    fetch = fetch or fetch_text
    series = []
    for t in tickers:
        url = STOOQ_URL.format(symbol=stooq_symbol(t))
        series.append(parse_stooq_csv(fetch(url), t))
    prices = pd.concat(series, axis=1).sort_index()
    prices.index.name = "Date"
    return prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily closes from Stooq.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out", default=str(DEFAULT_PRICES_PATH))
    parser.add_argument("--tickers", nargs="*", help="override the tickers in the config")
    args = parser.parse_args(argv)

    tickers = args.tickers or load_config(args.config).portfolio.tickers
    print(f"Downloading {len(tickers)} tickers from Stooq: {', '.join(tickers)}")
    prices = download_prices(tickers)
    path = save_prices_csv(prices, args.out)
    starts = prices.apply(pd.Series.first_valid_index)
    print(f"Saved {len(prices)} rows to {path}")
    print(f"Common history starts {starts.max().date()} (latest-starting: {starts.idxmax()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
