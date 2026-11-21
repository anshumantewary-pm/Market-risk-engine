import numpy as np
import pandas as pd
import pytest

from riskengine.risk.ewma import FilteredHistoricalSimulation, ewma_variance, ewma_volatility
from riskengine.risk.historical import HistoricalSimulation


def test_ewma_recursion_matches_hand_calculation():
    r = np.array([0.01, -0.02, 0.03, 0.0])
    lam = 0.9
    var, var_next = ewma_variance(r, lam, seed_days=2)
    seed = np.var(r[:2], ddof=1)
    expected = [seed]
    for x in r[:-1]:
        expected.append(lam * expected[-1] + (1 - lam) * x**2)
    np.testing.assert_allclose(var[:, 0], expected)
    assert var_next[0] == pytest.approx(lam * expected[-1] + (1 - lam) * r[-1] ** 2)


def test_ewma_variance_at_t_ignores_return_at_t():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 100)
    var_a, _ = ewma_variance(r)
    r2 = r.copy()
    r2[50] = 0.5  # huge shock on day 50
    var_b, _ = ewma_variance(r2)
    np.testing.assert_allclose(var_a[:51], var_b[:51])  # day 50's own variance unchanged
    assert var_b[51, 0] > var_a[51, 0]


def test_ewma_rejects_bad_lambda():
    with pytest.raises(ValueError):
        ewma_variance(np.zeros(10), lam=1.0)


def test_ewma_volatility_frame_shape():
    idx = pd.bdate_range("2020-01-01", periods=60)
    df = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (60, 2)), idx, ["A", "B"])
    vol = ewma_volatility(df)
    assert vol.shape == df.shape and (vol > 0).all().all()


def _regime_change(n_calm=450, n_hot=50, seed=2):
    rng = np.random.default_rng(seed)
    x = np.r_[rng.normal(0, 0.005, n_calm), rng.normal(0, 0.03, n_hot)]
    idx = pd.bdate_range("2018-01-01", periods=len(x))
    return pd.DataFrame({"A": x}, index=idx), pd.Series({"A": 1.0})


def test_fhs_reacts_faster_than_historical_after_volatility_jump():
    df, w = _regime_change()
    fhs = FilteredHistoricalSimulation().estimate(df, w, 0.99)
    hs = HistoricalSimulation().estimate(df, w, 0.99)
    assert fhs.var > hs.var
    assert fhs.es >= fhs.var
    assert fhs.diagnostics["asset_vol_now"][0] > 0.015


def test_fhs_is_close_to_historical_when_volatility_is_constant():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=2000)
    df = pd.DataFrame({"A": rng.normal(0, 0.01, 2000)}, index=idx)
    w = pd.Series({"A": 1.0})
    fhs = FilteredHistoricalSimulation().estimate(df, w, 0.99).var
    hs = HistoricalSimulation().estimate(df, w, 0.99).var
    assert fhs == pytest.approx(hs, rel=0.3)
