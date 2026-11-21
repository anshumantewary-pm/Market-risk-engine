"""Recompute risk for edited weights by calling the same library code the pipeline uses."""

from __future__ import annotations

import pandas as pd

from riskengine.attribution.components import attribution_table
from riskengine.config import RiskConfig
from riskengine.risk.registry import build_estimators, estimate_all, estimates_frame, latest_window

WHATIF_SIMS = 5_000  # fewer Monte Carlo simulations so the page stays responsive


def risk_config_from_metadata(meta: dict) -> RiskConfig:
    r = dict(meta["risk"])
    r["confidence_levels"] = tuple(r["confidence_levels"])
    r["horizons"] = tuple(r["horizons"])
    return RiskConfig(**r)


def normalise(weights: dict[str, float]) -> pd.Series:
    """Scale raw slider values so they sum to 1 (long-only)."""
    w = pd.Series(weights, dtype=float).clip(lower=0)
    total = w.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return w / total


def recompute(
    returns: pd.DataFrame, weights: pd.Series, notional: float, cfg: RiskConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-day VaR/ES for every method and the attribution table, for ``weights``."""
    window = latest_window(returns[list(weights.index)], cfg.window)
    est = estimate_all(
        window, weights, notional, build_estimators(cfg, n_sims=WHATIF_SIMS),
        cfg.confidence_levels, (1,),
    )
    attrib = attribution_table(window, weights, max(cfg.confidence_levels), notional)
    return estimates_frame(est), attrib
