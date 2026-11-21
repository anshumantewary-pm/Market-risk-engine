import numpy as np
import pandas as pd
import pytest

from riskengine.risk.base import Estimator, VarEs, empirical_var_es, prepare
from riskengine.risk.historical import HistoricalSimulation


def _single_asset(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=idx), pd.Series({"A": 1.0})


def test_empirical_var_es_on_hand_built_losses():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    var, es = empirical_var_es(losses, 0.95)
    assert var == pytest.approx(np.quantile(losses, 0.95))  # 95.05
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_historical_uses_losses_not_returns():
    # 100 losing days: returns from -0.1% down to -10%
    returns = -np.arange(1, 101) / 1000.0  # all losses, worst = 0.100
    df, w = _single_asset(returns)
    res = HistoricalSimulation().estimate(df, w, 0.99)
    assert res.var > 0
    assert res.var == pytest.approx(np.quantile(np.arange(1, 101) / 1000.0, 0.99))
    assert res.es >= res.var
    assert res.diagnostics["observations"] == 100


def test_historical_portfolio_combines_assets():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=250)
    df = pd.DataFrame(rng.normal(0, 0.01, (250, 2)), index=idx, columns=["A", "B"])
    w = pd.Series({"A": 0.3, "B": 0.7})
    res = HistoricalSimulation().estimate(df, w, 0.95)
    losses = -(0.3 * df["A"] + 0.7 * df["B"])
    assert res.var == pytest.approx(np.quantile(losses, 0.95))


def test_estimator_satisfies_protocol():
    assert isinstance(HistoricalSimulation(), Estimator)


def test_prepare_rejects_short_windows_and_nans():
    df, w = _single_asset(np.zeros(10))
    with pytest.raises(ValueError, match="at least"):
        prepare(df, w, 0.99)
    df, w = _single_asset(np.r_[np.zeros(40), np.nan])
    with pytest.raises(ValueError, match="NaN"):
        prepare(df, w, 0.99)


def test_var_es_result_rejects_es_below_var():
    with pytest.raises(ValueError):
        VarEs(0.05, 0.04)
