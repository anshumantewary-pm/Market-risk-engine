import math

import numpy as np
import pandas as pd
import pytest

from riskengine.config import RiskConfig
from riskengine.risk.base import Estimator
from riskengine.risk.registry import (
    METHOD_LABELS,
    build_estimators,
    estimate_all,
    estimates_frame,
    latest_window,
)


@pytest.fixture(scope="module")
def window():
    rng = np.random.default_rng(11)
    x = rng.standard_t(5, (500, 3)) * 0.01
    idx = pd.bdate_range("2019-01-01", periods=500)
    return pd.DataFrame(x, index=idx, columns=["A", "B", "C"]), pd.Series(
        {"A": 0.5, "B": 0.3, "C": 0.2}
    )


def test_registry_has_all_six_methods_with_labels():
    est = build_estimators(RiskConfig(mc_simulations=5000))
    assert list(est) == [
        "historical", "normal", "student_t", "ewma_fhs", "mc_normal", "mc_t_copula",
    ]
    assert all(isinstance(e, Estimator) for e in est.values())
    assert set(est) == set(METHOD_LABELS)


def test_n_sims_override_applies_to_monte_carlo():
    est = build_estimators(RiskConfig(), n_sims=2000)
    assert est["mc_normal"].n_sims == 2000 and est["mc_t_copula"].n_sims == 2000


def test_estimate_all_covers_every_combination_and_obeys_conventions(window):
    df, w = window
    est = build_estimators(RiskConfig(mc_simulations=5000))
    out = estimate_all(df, w, 1_000_000, est, (0.95, 0.99), (1, 10))
    assert len(out) == 6 * 2 * 2
    for e in out:
        assert e.es_pct >= e.var_pct > 0
        assert e.var_usd == pytest.approx(e.var_pct * 1_000_000)
        assert e.window_end == df.index[-1]


def test_ten_day_is_sqrt_ten_times_one_day(window):
    df, w = window
    est = {"historical": build_estimators()["historical"]}
    out = {(e.alpha, e.horizon_days): e for e in estimate_all(df, w, 1.0, est, (0.99,), (1, 10))}
    assert out[(0.99, 10)].var_pct == pytest.approx(out[(0.99, 1)].var_pct * math.sqrt(10))


def test_99_is_riskier_than_95(window):
    df, w = window
    est = build_estimators(RiskConfig(mc_simulations=5000))
    frame = estimates_frame(estimate_all(df, w, 1.0, est, (0.95, 0.99), (1,)))
    pivot = frame.pivot(index="method", columns="alpha", values="var_pct")
    assert (pivot[0.99] > pivot[0.95]).all()
    assert "diagnostics" not in frame.columns and "label" in frame.columns


def test_latest_window_takes_the_most_recent_rows(window):
    df, _ = window
    last = latest_window(df, 100)
    assert len(last) == 100 and last.index[-1] == df.index[-1]
    with pytest.raises(ValueError):
        latest_window(df, 1000)
