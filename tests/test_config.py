import copy

import pytest

from riskengine.config import ConfigError, RiskConfig, load_config, parse_config

BASE = {
    "portfolio": {
        "name": "Test",
        "notional": 1_000_000,
        "positions": {
            "AAA": {"weight": 0.6, "asset_class": "equity"},
            "BBB": {"weight": 0.4, "asset_class": "bonds"},
        },
    },
    "risk": {"window": 250, "confidence_levels": [0.95, 0.99]},
}


def _with(path, value):
    raw = copy.deepcopy(BASE)
    node = raw
    *parents, last = path
    for key in parents:
        node = node[key]
    node[last] = value
    return raw


def test_default_config_file_loads_and_is_balanced(repo_root):
    cfg = load_config(repo_root / "config" / "portfolio.yaml")
    assert len(cfg.portfolio.tickers) == 12
    assert cfg.portfolio.notional == 10_000_000
    assert cfg.portfolio.weight_series().sum() == pytest.approx(1.0)
    assert cfg.risk.window == 500
    assert cfg.risk.confidence_levels == (0.95, 0.99)


def test_parse_config_builds_weights_in_order():
    cfg = parse_config(BASE)
    w = cfg.portfolio.weight_series()
    assert list(w.index) == ["AAA", "BBB"]
    assert w["AAA"] == 0.6
    assert cfg.portfolio.asset_classes["BBB"] == "bonds"
    assert cfg.risk.window == 250
    assert cfg.risk.horizons == (1, 10)  # default kept


def test_weights_must_sum_to_one():
    raw = _with(("portfolio", "positions", "BBB", "weight"), 0.3)
    with pytest.raises(ConfigError, match="sum to 1"):
        parse_config(raw)


def test_notional_must_be_positive():
    with pytest.raises(ConfigError, match="notional"):
        parse_config(_with(("portfolio", "notional"), -5))


def test_missing_portfolio_section_is_rejected():
    with pytest.raises(ConfigError, match="portfolio"):
        parse_config({"risk": {}})


def test_unknown_risk_setting_is_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        parse_config(_with(("risk", "windw"), 100))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 10},
        {"confidence_levels": (0.4,)},
        {"horizons": (0,)},
        {"ewma_lambda": 1.0},
        {"mc_simulations": 10},
    ],
)
def test_risk_config_validates_ranges(kwargs):
    with pytest.raises(ConfigError):
        RiskConfig(**kwargs)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
