"""Downloader tests use a fixture file and a fake fetcher: no network access."""

import pandas as pd
import pytest

from riskengine.download import (
    DownloadError,
    download_prices,
    main,
    parse_stooq_csv,
    stooq_symbol,
)


@pytest.fixture
def spy_csv(repo_root):
    return (repo_root / "tests" / "fixtures" / "stooq_spy.csv").read_text()


def test_stooq_symbol():
    assert stooq_symbol("SPY") == "spy.us"
    assert stooq_symbol(" qqq ") == "qqq.us"


def test_parse_fixture(spy_csv):
    s = parse_stooq_csv(spy_csv, "SPY")
    assert s.name == "SPY"
    assert len(s) == 4
    assert s.index[0] == pd.Timestamp("2024-01-02")
    assert s.iloc[-1] == pytest.approx(467.92)


def test_parse_sorts_and_deduplicates(spy_csv):
    header, *rows = spy_csv.strip().splitlines()
    shuffled = "\n".join([header, rows[2], rows[0], rows[1], rows[3], rows[3]])
    s = parse_stooq_csv(shuffled, "SPY")
    assert s.index.is_monotonic_increasing
    assert not s.index.has_duplicates


@pytest.mark.parametrize("body", ["No data", "", "<html>blocked</html>"])
def test_parse_rejects_non_csv_bodies(body):
    with pytest.raises(DownloadError):
        parse_stooq_csv(body, "ZZZ")


def test_download_merges_tickers_with_different_histories(spy_csv):
    short = "Date,Open,High,Low,Close,Volume\n2024-01-04,1,1,1,10.0,1\n2024-01-05,1,1,1,11.0,1\n"
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return spy_csv if "spy.us" in url else short

    prices = download_prices(["SPY", "NEW"], fetch=fake_fetch)
    assert len(calls) == 2 and "s=new.us" in calls[1]
    assert list(prices.columns) == ["SPY", "NEW"]
    assert prices.index.name == "Date"
    assert prices["NEW"].isna().sum() == 2  # starts later; the loader trims this


def test_main_writes_the_prices_file(tmp_path, monkeypatch, spy_csv):
    monkeypatch.setattr("riskengine.download.fetch_text", lambda url: spy_csv)
    out = tmp_path / "prices.csv"
    assert main(["--tickers", "SPY", "--out", str(out)]) == 0
    assert out.read_text().splitlines()[0] == "Date,SPY"
