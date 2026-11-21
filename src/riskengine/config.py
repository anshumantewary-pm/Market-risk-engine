"""Typed configuration for the portfolio and the risk run.

The YAML file (``config/portfolio.yaml``) is parsed into frozen dataclasses and validated
once, so the rest of the engine can trust what it receives.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from riskengine.conventions import (
    DEFAULT_CONFIDENCE_LEVELS,
    DEFAULT_HORIZONS,
    DEFAULT_WINDOW,
    validate_alpha,
)

DEFAULT_CONFIG_PATH = Path("config/portfolio.yaml")
WEIGHT_SUM_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or has invalid values."""


@dataclass(frozen=True)
class PortfolioConfig:
    """Positions held: ticker weights, their asset classes and the dollar notional."""

    name: str
    notional: float
    weights: Mapping[str, float]
    asset_classes: Mapping[str, str]

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)

    def weight_series(self) -> pd.Series:
        """Weights as a Series indexed by ticker, in config order."""
        return pd.Series(dict(self.weights), name="weight", dtype=float)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigError("portfolio must have at least one position")
        if not (math.isfinite(self.notional) and self.notional > 0):
            raise ConfigError(f"notional must be a positive number, got {self.notional!r}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ConfigError(f"weights must sum to 1, got {total:.8f}")
        missing = set(self.weights) - set(self.asset_classes)
        if missing:
            raise ConfigError(f"asset_class missing for: {sorted(missing)}")


@dataclass(frozen=True)
class RiskConfig:
    """Settings for estimation, backtesting and simulation."""

    window: int = DEFAULT_WINDOW
    confidence_levels: tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    ewma_lambda: float = 0.94
    mc_simulations: int = 20_000
    backtest_alpha: float = 0.99
    traffic_light_days: int = 250
    seed: int = 42

    def __post_init__(self) -> None:
        if self.window < 30:
            raise ConfigError(f"window must be at least 30 days, got {self.window}")
        try:
            for a in (*self.confidence_levels, self.backtest_alpha):
                validate_alpha(a)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if any(int(h) != h or h < 1 for h in self.horizons):
            raise ConfigError(f"horizons must be positive whole days, got {self.horizons}")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ConfigError(f"ewma_lambda must be in (0, 1), got {self.ewma_lambda}")
        if self.mc_simulations < 1_000:
            raise ConfigError(f"mc_simulations must be at least 1000, got {self.mc_simulations}")
        if self.traffic_light_days < 1:
            raise ConfigError("traffic_light_days must be positive")


@dataclass(frozen=True)
class Config:
    portfolio: PortfolioConfig
    risk: RiskConfig = field(default_factory=RiskConfig)


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Build a validated :class:`Config` from a plain dictionary (e.g. parsed YAML)."""
    if not isinstance(raw, Mapping) or "portfolio" not in raw:
        raise ConfigError("config must contain a 'portfolio' section")
    p = raw["portfolio"]
    positions = p.get("positions") or {}
    if not isinstance(positions, Mapping):
        raise ConfigError("portfolio.positions must be a mapping of ticker -> settings")

    weights: dict[str, float] = {}
    classes: dict[str, str] = {}
    for ticker, spec in positions.items():
        if not isinstance(spec, Mapping) or "weight" not in spec:
            raise ConfigError(f"position {ticker!r} needs a 'weight'")
        weights[str(ticker)] = float(spec["weight"])
        classes[str(ticker)] = str(spec.get("asset_class", "other"))

    portfolio = PortfolioConfig(
        name=str(p.get("name", "Portfolio")),
        notional=float(p.get("notional", 0)),
        weights=weights,
        asset_classes=classes,
    )

    r = dict(raw.get("risk") or {})
    known = set(RiskConfig.__dataclass_fields__)
    unknown = set(r) - known
    if unknown:
        raise ConfigError(f"unknown risk settings: {sorted(unknown)}")
    for key in ("confidence_levels", "horizons"):
        if key in r:
            r[key] = tuple(r[key])
    return Config(portfolio=portfolio, risk=RiskConfig(**r))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return parse_config(raw)
