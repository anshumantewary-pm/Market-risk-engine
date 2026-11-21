import math

import numpy as np
import pandas as pd
import pytest

from riskengine.backtest.summary import summary_table
from riskengine.backtest.tests import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    traffic_light,
    transition_counts,
)


def _series(t, positions):
    e = np.zeros(t, dtype=bool)
    e[list(positions)] = True
    return e


def test_kupiec_matches_hand_calculation():
    t, x, p = 250, 5, 0.01
    e = _series(t, range(0, 250, 50))
    expected = -2 * ((t - x) * math.log(1 - p) + x * math.log(p)) + 2 * (
        (t - x) * math.log(1 - x / t) + x * math.log(x / t)
    )
    res = kupiec_pof(e, 0.99)
    assert res.statistic == pytest.approx(expected)
    assert res.details["exceptions"] == 5 and res.dof == 1


def test_kupiec_exact_expected_rate_is_not_rejected():
    res = kupiec_pof(_series(1000, range(0, 1000, 100)), 0.99)  # 10 of 1000 = 1%
    assert res.statistic == pytest.approx(0.0, abs=1e-12)
    assert res.p_value == pytest.approx(1.0)
    assert not res.reject_at_5pct


def test_kupiec_rejects_too_many_exceptions():
    res = kupiec_pof(_series(1000, range(0, 1000, 33)), 0.99)  # ~3% instead of 1%
    assert res.reject_at_5pct


def test_kupiec_handles_zero_exceptions():
    res = kupiec_pof(np.zeros(1000, dtype=bool), 0.99)
    assert res.statistic == pytest.approx(-2 * 1000 * math.log(0.99))
    assert math.isfinite(res.p_value) and res.reject_at_5pct  # too few is also wrong


def test_transition_counts():
    c = transition_counts([0, 1, 1, 0, 0, 1])
    assert c == {"n00": 1, "n01": 2, "n10": 1, "n11": 1}


def test_christoffersen_rejects_clustered_exceptions():
    clustered = _series(1000, [*range(100, 105), *range(600, 605)])  # 10 exceptions in 2 runs
    res = christoffersen_independence(clustered)
    assert res.details["pi11"] > res.details["pi01"]
    assert res.reject_at_5pct


def test_christoffersen_accepts_spread_out_exceptions():
    spread = _series(1000, range(50, 1000, 100))
    res = christoffersen_independence(spread)
    assert res.details["n11"] == 0
    assert not res.reject_at_5pct


def test_conditional_coverage_is_the_sum_with_two_dof():
    e = _series(1000, [*range(100, 105), *range(600, 605)])
    cc = conditional_coverage(e, 0.99)
    assert cc.statistic == pytest.approx(
        kupiec_pof(e, 0.99).statistic + christoffersen_independence(e).statistic
    )
    assert cc.dof == 2


@pytest.mark.parametrize("x, zone", [(0, "green"), (4, "green"), (5, "yellow"),
                                     (9, "yellow"), (10, "red"), (15, "red")])
def test_traffic_light_basel_boundaries(x, zone):
    tl = traffic_light(_series(250, range(x)), alpha=0.99, days=250)
    assert tl.zone == zone
    assert (tl.green_max, tl.yellow_max) == (4, 9)


def test_traffic_light_only_counts_the_last_250_days():
    e = _series(500, range(0, 20))  # 20 old exceptions, none in the last 250 days
    tl = traffic_light(e, 0.99, 250)
    assert tl.exceptions == 0 and tl.zone == "green"


def test_empty_series_is_rejected():
    with pytest.raises(ValueError):
        kupiec_pof([], 0.99)


def test_summary_table_has_one_row_per_method():
    idx = pd.bdate_range("2020-01-01", periods=300)
    f = pd.DataFrame(
        {"var": 0.02, "es": 0.03, "realised_loss": 0.0, "exception": _series(300, [10, 200])},
        index=idx,
    )
    table = summary_table({"a": f, "b": f}, alpha=0.99)
    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "exceptions"] == 2
    assert table.loc["a", "expected_exceptions"] == pytest.approx(3.0)
    assert table.loc["a", "traffic_light"] == "green"
    assert {"kupiec_p", "christoffersen_p", "cc_p"} <= set(table.columns)
