import numpy as np
import pandas as pd
import pytest

from riskengine.risk.montecarlo import (
    MonteCarloNormal,
    MonteCarloTCopula,
    empirical_quantiles,
    rank_correlation,
)
from riskengine.risk.parametric import ParametricNormal


@pytest.fixture(scope="module")
def normal_window():
    rng = np.random.default_rng(10)
    cov = np.array([[1.0, 0.6], [0.6, 2.0]]) * 1e-4
    x = rng.multivariate_normal([0.0, 0.0], cov, size=2000)
    idx = pd.bdate_range("2015-01-01", periods=len(x))
    return pd.DataFrame(x, index=idx, columns=["A", "B"]), pd.Series({"A": 0.4, "B": 0.6})


def test_mc_normal_converges_to_parametric_normal(normal_window):
    df, w = normal_window
    mc = MonteCarloNormal(n_sims=200_000).estimate(df, w, 0.99)
    pn = ParametricNormal().estimate(df, w, 0.99)
    assert mc.var == pytest.approx(pn.var, rel=0.02)
    assert mc.es == pytest.approx(pn.es, rel=0.03)


def test_mc_is_reproducible_with_the_same_seed(normal_window):
    df, w = normal_window
    a = MonteCarloTCopula(n_sims=5000, seed=1).estimate(df, w, 0.99)
    b = MonteCarloTCopula(n_sims=5000, seed=1).estimate(df, w, 0.99)
    c = MonteCarloTCopula(n_sims=5000, seed=2).estimate(df, w, 0.99)
    assert a == b
    assert a.var != c.var


def test_empirical_quantiles_match_numpy():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(300, 3))
    u = rng.random((1000, 3))
    ref = np.column_stack([np.quantile(r[:, j], u[:, j]) for j in range(3)])
    np.testing.assert_allclose(empirical_quantiles(r, u), ref, atol=1e-12)


def test_rank_correlation_recovers_normal_correlation():
    rng = np.random.default_rng(5)
    x = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=50_000)
    assert rank_correlation(x)[0, 1] == pytest.approx(0.7, abs=0.01)


def test_t_copula_keeps_marginals_inside_historical_range(normal_window):
    df, w = normal_window
    res = MonteCarloTCopula(n_sims=10_000).estimate(df, w, 0.99)
    worst_hist_loss = -(df @ w).min()
    worst_possible = (-df.min() * w).sum()  # every asset at its worst day at once
    assert 0 < res.var <= res.es <= worst_possible
    assert res.es <= worst_hist_loss * 2


def test_t_copula_has_more_tail_risk_than_normal_copula_for_same_marginals():
    # Same normal marginals; the t copula should put more weight on joint crashes.
    rng = np.random.default_rng(6)
    x = rng.multivariate_normal([0, 0, 0], np.full((3, 3), 0.5) + 0.5 * np.eye(3), 3000) * 0.01
    idx = pd.bdate_range("2010-01-01", periods=len(x))
    df = pd.DataFrame(x, index=idx, columns=list("ABC"))
    w = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    t_es = MonteCarloTCopula(n_sims=50_000, dof=3).estimate(df, w, 0.99).es
    gauss_like_es = MonteCarloTCopula(n_sims=50_000, dof=200).estimate(df, w, 0.99).es
    assert t_es > gauss_like_es


def test_t_copula_rejects_low_dof():
    with pytest.raises(ValueError):
        MonteCarloTCopula(dof=2)
