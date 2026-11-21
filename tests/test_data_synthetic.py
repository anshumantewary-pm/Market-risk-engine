import numpy as np
import pandas as pd
import pytest

from riskengine.data.load import validate_prices
from riskengine.data.returns import log_returns
from riskengine.data.synthetic import ASSETS, EPISODES, correlation_matrix, generate_prices


@pytest.fixture(scope="module")
def prices():
    return generate_prices(seed=7)


def test_correlation_matrix_is_valid():
    c = correlation_matrix(list(ASSETS))
    np.testing.assert_allclose(np.diag(c), 1.0)
    np.testing.assert_allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > 0


def test_unknown_ticker_is_rejected():
    with pytest.raises(ValueError, match="ZZZ"):
        correlation_matrix(["SPY", "ZZZ"])


def test_output_matches_the_prices_file_contract(prices):
    assert list(prices.columns) == list(ASSETS)
    assert prices.index[0] == pd.Timestamp("2005-01-03")
    assert prices.index[-1] <= pd.Timestamp("2024-12-31")
    assert (prices.index.dayofweek < 5).all()
    clean, report = validate_prices(prices)
    assert report.dropped_rows == 0 and report.trimmed_rows == 0


def test_same_seed_same_data_different_seed_different_data():
    a = generate_prices(["SPY", "TLT"], end="2006-12-31", seed=1)
    b = generate_prices(["SPY", "TLT"], end="2006-12-31", seed=1)
    c = generate_prices(["SPY", "TLT"], end="2006-12-31", seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_returns_have_fat_tails_and_sensible_correlations(prices):
    r = log_returns(prices)
    assert (r.kurt() > 1.0).all()  # excess kurtosis: fatter tails than normal
    corr = r.corr()
    assert corr.loc["SPY", "QQQ"] > 0.7
    assert corr.loc["SPY", "TLT"] < 0
    assert 0.1 < r["IEF"].std() * np.sqrt(252) / (r["TLT"].std() * np.sqrt(252)) < 0.9


def test_stress_episode_is_a_drawdown_for_equities(prices):
    ep = EPISODES[0]
    window = prices.loc[ep.start:].iloc[: ep.days]
    assert window["SPY"].iloc[-1] < window["SPY"].iloc[0]
