"""Build every estimator from config and produce the headline VaR/ES table."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from riskengine.config import RiskConfig
from riskengine.conventions import scale_horizon
from riskengine.risk.base import Estimator
from riskengine.risk.ewma import FilteredHistoricalSimulation
from riskengine.risk.historical import HistoricalSimulation
from riskengine.risk.montecarlo import MonteCarloNormal, MonteCarloTCopula
from riskengine.risk.parametric import ParametricNormal, ParametricStudentT

METHOD_LABELS = {
    "historical": "Historical simulation",
    "normal": "Parametric normal",
    "student_t": "Parametric Student-t",
    "ewma_fhs": "EWMA filtered HS",
    "mc_normal": "Monte Carlo (normal)",
    "mc_t_copula": "Monte Carlo (t-copula)",
}


def build_estimators(
    cfg: RiskConfig | None = None, n_sims: int | None = None
) -> dict[str, Estimator]:
    """All estimators, keyed by name, configured from ``cfg``.

    ``n_sims`` overrides the Monte Carlo simulation count (the backtest uses fewer
    simulations than the headline numbers to keep run time reasonable).
    """
    cfg = cfg or RiskConfig()
    sims = n_sims or cfg.mc_simulations
    estimators: list[Estimator] = [
        HistoricalSimulation(),
        ParametricNormal(),
        ParametricStudentT(),
        FilteredHistoricalSimulation(lam=cfg.ewma_lambda),
        MonteCarloNormal(n_sims=sims, seed=cfg.seed),
        MonteCarloTCopula(n_sims=sims, seed=cfg.seed),
    ]
    return {e.name: e for e in estimators}


@dataclass(frozen=True)
class RiskEstimate:
    """One VaR/ES figure, in both % of notional and dollars."""

    method: str
    alpha: float
    horizon_days: int
    var_pct: float
    es_pct: float
    var_usd: float
    es_usd: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    diagnostics: dict[str, Any] = field(default_factory=dict)


def latest_window(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """The most recent ``window`` rows: the data available for tomorrow's forecast."""
    if len(returns) < window:
        raise ValueError(f"need {window} days of returns, have {len(returns)}")
    return returns.iloc[-window:]


def estimate_all(
    window_returns: pd.DataFrame,
    weights: pd.Series,
    notional: float,
    estimators: dict[str, Estimator],
    confidence_levels: tuple[float, ...],
    horizons: tuple[int, ...],
) -> list[RiskEstimate]:
    """Every method × confidence level × horizon on one window.

    Each method is fitted once per confidence level at 1 day; longer horizons are scaled by
    the square root of time (see ``riskengine.conventions``).
    """
    out = []
    start, end = window_returns.index[0], window_returns.index[-1]
    for name, est in estimators.items():
        for alpha in confidence_levels:
            res = est.estimate(window_returns, weights, alpha)
            for h in horizons:
                var, es = scale_horizon(res.var, h), scale_horizon(res.es, h)
                out.append(
                    RiskEstimate(
                        method=name, alpha=alpha, horizon_days=h,
                        var_pct=float(var), es_pct=float(es),
                        var_usd=float(var * notional), es_usd=float(es * notional),
                        window_start=start, window_end=end, diagnostics=res.diagnostics,
                    )
                )
    return out


def estimates_frame(estimates: list[RiskEstimate]) -> pd.DataFrame:
    """Flatten estimates into a table (diagnostics dropped)."""
    rows = []
    for e in estimates:
        row = asdict(e)
        row.pop("diagnostics")
        row["label"] = METHOD_LABELS.get(e.method, e.method)
        rows.append(row)
    return pd.DataFrame(rows)
