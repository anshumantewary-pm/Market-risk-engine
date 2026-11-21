import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine.risk.parametric import (
    ParametricNormal,
    ParametricStudentT,
    fit_t_dof,
    normal_var_es,
    portfolio_moments,
    student_t_var_es,
)


def _frame(x, cols=("A",)):
    x = np.atleast_2d(np.asarray(x).T).T
    idx = pd.bdate_range("2000-01-03", periods=len(x))
    return pd.DataFrame(x, index=idx, columns=list(cols))


def test_normal_closed_form_known_values():
    # Standard normal, 99%: VaR = 2.3263, ES = 2.6652
    var, es = normal_var_es(0.0, 1.0, 0.99)
    assert var == pytest.approx(2.326348, abs=1e-6)
    assert es == pytest.approx(2.665214, abs=1e-6)
    # A positive mean reduces both by μ
    var2, es2 = normal_var_es(0.001, 1.0, 0.99)
    assert var2 == pytest.approx(var - 0.001)
    assert es2 == pytest.approx(es - 0.001)


def test_portfolio_moments_match_matrix_formula():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, (400, 3))
    w = np.array([0.5, 0.3, 0.2])
    mu, sigma = portfolio_moments(r, w)
    assert mu == pytest.approx(r.mean(0) @ w)
    assert sigma == pytest.approx(np.std(r @ w, ddof=1))  # same thing, computed directly


def test_normal_estimator_recovers_true_var_on_large_normal_sample():
    rng = np.random.default_rng(2)
    df = _frame(rng.normal(0.0, 0.01, 200_000))
    res = ParametricNormal().estimate(df, pd.Series({"A": 1.0}), 0.99)
    assert res.var == pytest.approx(0.01 * 2.326348, rel=0.01)
    assert res.es == pytest.approx(0.01 * 2.665214, rel=0.01)


def test_student_t_closed_form_matches_scipy_quantile_and_numeric_es():
    nu, sigma, alpha = 5.0, 0.02, 0.99
    var, es = student_t_var_es(0.0, sigma, nu, alpha)
    s = sigma * np.sqrt((nu - 2) / nu)
    assert var == pytest.approx(s * stats.t.ppf(alpha, nu))
    # ES = E[L | L >= VaR], by numerical integration of the t density
    dist = stats.t(nu, scale=s)
    numeric_es = dist.expect(lambda x: x, lb=var, conditional=True)
    assert es == pytest.approx(numeric_es, rel=1e-6)


def test_student_t_approaches_normal_for_large_nu():
    var_t, es_t = student_t_var_es(0.0, 1.0, 1e6, 0.99)
    var_n, es_n = normal_var_es(0.0, 1.0, 0.99)
    assert var_t == pytest.approx(var_n, rel=1e-4)
    assert es_t == pytest.approx(es_n, rel=1e-4)


def test_student_t_has_fatter_99pct_tail_than_normal_at_same_variance():
    var_t, es_t = student_t_var_es(0.0, 1.0, 4.0, 0.99)
    var_n, es_n = normal_var_es(0.0, 1.0, 0.99)
    assert es_t > es_n


def test_fit_t_dof_recovers_nu():
    rng = np.random.default_rng(3)
    assert fit_t_dof(rng.standard_t(4, 20_000)) == pytest.approx(4, abs=0.6)
    assert fit_t_dof(rng.normal(size=20_000)) > 30


def test_student_t_estimator_reports_nu_and_es_above_var():
    rng = np.random.default_rng(4)
    df = _frame(rng.standard_t(4, (500, 2)) * 0.01, cols=("A", "B"))
    res = ParametricStudentT().estimate(df, pd.Series({"A": 0.5, "B": 0.5}), 0.99)
    assert 2.1 <= res.diagnostics["nu"] <= 200
    assert res.es > res.var > 0
