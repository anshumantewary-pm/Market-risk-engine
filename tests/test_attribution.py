import numpy as np
import pandas as pd
import pytest

from riskengine.attribution.components import (
    attribution_table,
    historical_component_es,
    parametric_components,
)
from riskengine.risk.historical import HistoricalSimulation
from riskengine.risk.parametric import ParametricNormal


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(31)
    cov = np.array([[4.0, 1.0, -0.5], [1.0, 2.0, 0.0], [-0.5, 0.0, 1.0]]) * 1e-4
    x = rng.multivariate_normal(np.zeros(3), cov, size=750)
    idx = pd.bdate_range("2018-01-01", periods=750)
    df = pd.DataFrame(x, index=idx, columns=["EQ", "CR", "TSY"])
    return df, pd.Series({"EQ": 0.5, "CR": 0.3, "TSY": 0.2})


def test_component_var_sums_to_total_parametric_var(data):
    df, w = data
    comp = parametric_components(df, w, 0.99)
    pn = ParametricNormal().estimate(df, w, 0.99)
    zero_mean_var = pn.var + pn.diagnostics["mu"]  # VaR = z·σ − μ  ->  z·σ
    assert comp["component_var"].sum() == pytest.approx(zero_mean_var, rel=1e-10)
    assert comp["risk_share"].sum() == pytest.approx(1.0, rel=1e-10)


def test_marginal_var_matches_finite_difference(data):
    df, w = data
    comp = parametric_components(df, w, 0.99)

    def total(weights):
        return parametric_components(df, weights, 0.99)["component_var"].sum()

    h = 1e-6
    bumped = w.copy()
    bumped["EQ"] += h
    numeric = (total(bumped) - total(w)) / h
    assert comp.loc["EQ", "marginal_var"] == pytest.approx(numeric, rel=1e-4)


def test_hedge_has_negative_component(data):
    df, _ = data
    # TSY is negatively correlated with EQ; a large EQ book makes TSY a hedge
    w = pd.Series({"EQ": 0.9, "CR": 0.0, "TSY": 0.1})
    comp = parametric_components(df, w, 0.99)
    assert comp.loc["TSY", "component_var"] < 0


def test_component_es_sums_to_historical_es(data):
    df, w = data
    ces = historical_component_es(df, w, 0.975)
    hs = HistoricalSimulation().estimate(df, w, 0.975)
    assert ces.sum() == pytest.approx(hs.es, rel=1e-12)


def test_attribution_table_columns_and_concentration(data):
    df, w = data
    t = attribution_table(df, w, 0.99, 1_000_000, {"EQ": "equity", "CR": "credit"})
    assert t.index.name == "ticker"
    assert t.loc["TSY", "asset_class"] == "other"
    assert t["weight_share"].sum() == pytest.approx(1.0)
    assert t["concentration"].sum() == pytest.approx(0.0, abs=1e-12)
    assert t.loc["EQ", "concentration"] > 0  # the volatile asset carries extra risk
    np.testing.assert_allclose(t["component_var_usd"], t["component_var"] * 1_000_000)
