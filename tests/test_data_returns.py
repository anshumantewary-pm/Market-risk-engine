import numpy as np
import pandas as pd
import pytest

from riskengine.data.returns import log_returns, portfolio_pnl, portfolio_returns


@pytest.fixture
def prices():
    idx = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame({"AAA": [100.0, 110.0, 99.0, 99.0], "BBB": [50.0, 50.0, 55.0, 44.0]}, idx)


def test_log_returns_match_hand_calculation(prices):
    r = log_returns(prices)
    assert len(r) == 3
    assert r.index[0] == prices.index[1]
    assert r.loc[prices.index[1], "AAA"] == pytest.approx(np.log(1.1))
    assert r.loc[prices.index[3], "BBB"] == pytest.approx(np.log(0.8))
    assert r.loc[prices.index[3], "AAA"] == 0.0


def test_log_returns_reject_non_positive_prices(prices):
    with pytest.raises(ValueError):
        log_returns(prices.assign(AAA=0.0))


def test_portfolio_return_is_weighted_sum(prices):
    r = log_returns(prices)
    w = pd.Series({"AAA": 0.25, "BBB": 0.75})
    rp = portfolio_returns(r, w)
    expected = 0.25 * r["AAA"] + 0.75 * r["BBB"]
    pd.testing.assert_series_equal(rp, expected, check_names=False)
    assert rp.name == "portfolio_return"


def test_weights_order_does_not_matter(prices):
    r = log_returns(prices)
    a = portfolio_returns(r, pd.Series({"AAA": 0.4, "BBB": 0.6}))
    b = portfolio_returns(r, pd.Series({"BBB": 0.6, "AAA": 0.4}))
    pd.testing.assert_series_equal(a, b)


def test_missing_ticker_in_returns_raises(prices):
    with pytest.raises(ValueError, match="CCC"):
        portfolio_returns(log_returns(prices), pd.Series({"AAA": 0.5, "CCC": 0.5}))


def test_pnl_scales_by_notional(prices):
    r = log_returns(prices)
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    pnl = portfolio_pnl(r, w, 1_000_000)
    np.testing.assert_allclose(pnl.values, 1_000_000 * portfolio_returns(r, w).values)
    assert pnl.name == "pnl"
