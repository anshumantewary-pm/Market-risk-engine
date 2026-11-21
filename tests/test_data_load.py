import numpy as np
import pandas as pd
import pytest

from riskengine.data.load import (
    DataError,
    load_prices,
    read_prices_csv,
    save_prices_csv,
    validate_prices,
)


def _prices(n=10, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"AAA": np.linspace(100, 110, n), "BBB": np.linspace(50, 45, n)}, index=idx
    )


def test_valid_prices_pass_through_unchanged():
    df = _prices()
    clean, report = validate_prices(df)
    pd.testing.assert_frame_equal(clean, df)
    assert report.n_rows == 10
    assert report.trimmed_rows == report.dropped_rows == 0
    assert report.start == df.index[0] and report.end == df.index[-1]


def test_restricts_to_common_range_when_a_ticker_starts_late():
    df = _prices()
    df.iloc[:3, 1] = np.nan  # BBB starts three days late
    clean, report = validate_prices(df)
    assert clean.index[0] == df.index[3]
    assert report.trimmed_rows == 3
    assert report.dropped_rows == 0


def test_interior_gap_is_dropped_and_reported_not_filled():
    df = _prices()
    df.iloc[5, 0] = np.nan
    clean, report = validate_prices(df)
    assert df.index[5] not in clean.index
    assert report.dropped_rows == 1
    assert not clean.isna().any().any()


def test_selects_requested_tickers_in_order():
    clean, report = validate_prices(_prices(), tickers=["BBB"])
    assert list(clean.columns) == ["BBB"]
    assert report.tickers == ("BBB",)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.iloc[::-1], "sorted"),
        (lambda d: pd.concat([d, d.iloc[[0]]]).sort_index(), "duplicate"),
        (lambda d: d.assign(AAA=-1.0), "non-positive"),
        (lambda d: d.assign(AAA="x"), "non-numeric"),
        (lambda d: d.assign(BBB=np.nan), "no prices at all"),
    ],
)
def test_contract_violations_raise(mutate, message):
    with pytest.raises(DataError, match=message):
        validate_prices(mutate(_prices()))


def test_missing_ticker_raises():
    with pytest.raises(DataError, match="missing"):
        validate_prices(_prices(), tickers=["AAA", "ZZZ"])


def test_long_calendar_gap_raises():
    df = _prices(20)
    df = df.drop(df.index[5:15])  # two-week hole
    with pytest.raises(DataError, match="gap"):
        validate_prices(df, max_gap_days=10)


def test_save_and_load_round_trip(tmp_path):
    df = _prices()
    path = save_prices_csv(df, tmp_path / "raw" / "prices.csv")
    assert path.read_text().splitlines()[0] == "Date,AAA,BBB"
    loaded, _ = load_prices(path)
    assert loaded.index.name == "Date"
    pd.testing.assert_frame_equal(
        loaded, df, check_freq=False, check_names=False, check_index_type=False, atol=1e-6
    )


def test_missing_file_explains_how_to_get_data(tmp_path):
    with pytest.raises(DataError, match="riskengine.download"):
        read_prices_csv(tmp_path / "missing.csv")
