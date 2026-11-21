import numpy as np
import pandas as pd
import pytest

from riskengine.config import (
    ConfigError,
    FactorShock,
    HistoricalWindow,
    load_stress_config,
    parse_stress_config,
)
from riskengine.stress.historical import replay_all, replay_window
from riskengine.stress.hypothetical import ILLUSTRATIVE, apply_shock, position_shocks
from riskengine.stress.result import stress_frame


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = [100, 100, 100, 90, 80, 85, 88, 90, 90, 90]  # falls 20% then recovers to -10%
    b = [50] * 10                                     # flat
    return pd.DataFrame({"A": a, "B": b}, index=idx, dtype=float)


W = pd.Series({"A": 0.6, "B": 0.4})


def _window(prices, i, j, name="w"):
    return HistoricalWindow(name, prices.index[i], prices.index[j])


def test_replay_matches_hand_calculation(prices):
    # Window from day 3 to day 7: base = close on day 2 (100), end = day 7 (90) -> A -10%
    res = replay_window(prices, W, 1_000_000, _window(prices, 3, 7))
    assert res.status == "run"
    assert res.by_position_usd["A"] == pytest.approx(1_000_000 * 0.6 * -0.10)
    assert res.by_position_usd["B"] == pytest.approx(0.0)
    assert res.pnl_pct == pytest.approx(-0.06)
    # Worst day: 90 -> 80 on day 4, log move ln(80/90) on a 60% position
    assert res.worst_day == prices.index[4]
    assert res.worst_day_usd == pytest.approx(1_000_000 * 0.6 * np.log(80 / 90))


def test_window_not_covered_by_data_is_skipped_with_reason(prices):
    early = HistoricalWindow("too early", pd.Timestamp("2019-06-01"), pd.Timestamp("2020-01-03"))
    late = HistoricalWindow("too late", prices.index[5], pd.Timestamp("2021-01-01"))
    for res in replay_all(prices, W, 1.0, [early, late]):
        assert res.status == "skipped"
        assert "data" in res.reason
        assert np.isnan(res.pnl_usd)


def test_position_shocks_ticker_overrides_asset_class():
    sc = FactorShock("x", {"equity": -0.2, "B": 0.05})
    s = position_shocks(["A", "B", "C"], {"A": "equity", "B": "equity"}, sc)
    assert s.to_dict() == {"A": -0.2, "B": 0.05, "C": 0.0}


def test_hypothetical_shock_pnl_and_label():
    sc = FactorShock("Equity sell-off", {"equity": -0.20, "bonds": 0.03})
    res = apply_shock(W, {"A": "equity", "B": "bonds"}, 10_000_000, sc)
    assert res.pnl_usd == pytest.approx(10_000_000 * (0.6 * -0.20 + 0.4 * 0.03))
    assert res.reason == ILLUSTRATIVE and res.kind == "hypothetical"


def test_stress_frame_spreads_positions(prices):
    res = [replay_window(prices, W, 1.0, _window(prices, 3, 7, "r"))]
    df = stress_frame(res)
    assert {"name", "kind", "status", "pnl_usd", "pnl_A", "pnl_B"} <= set(df.columns)


def test_default_scenario_file_loads(repo_root):
    cfg = load_stress_config(repo_root / "config" / "stress_scenarios.yaml")
    assert len(cfg.historical) == 3 and len(cfg.hypothetical) >= 3
    assert all(h.start < h.end for h in cfg.historical)


@pytest.mark.parametrize(
    "raw",
    [
        {"historical": [{"name": "x", "start": "2020-02-01", "end": "2020-01-01"}]},
        {"historical": [{"name": "x", "start": "2020-01-01"}]},
        {"hypothetical": [{"name": "x", "shocks": {}}]},
        {"hypothetical": [{"name": "x", "shocks": {"A": -1.5}}]},
    ],
)
def test_bad_scenarios_are_rejected(raw):
    with pytest.raises(ConfigError):
        parse_stress_config(raw)
