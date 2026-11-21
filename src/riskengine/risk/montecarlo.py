"""Monte Carlo VaR and ES from simulated one-day asset returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.risk.base import VarEs, empirical_var_es, prepare

DEFAULT_SIMULATIONS = 20_000


def _cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor with a tiny diagonal nudge if the matrix is only semi-definite."""
    cov = np.atleast_2d(cov)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.trace(cov) / len(cov)
        return np.linalg.cholesky(cov + jitter * np.eye(len(cov)))


class MonteCarloNormal:
    """Simulate asset returns from a multivariate normal with the window's mean and covariance.

    The same seed is used on every call, so day-to-day changes in the estimate come from
    the data, not from simulation noise (common random numbers).
    """

    name = "mc_normal"

    def __init__(self, n_sims: int = DEFAULT_SIMULATIONS, seed: int = 42) -> None:
        self.n_sims = n_sims
        self.seed = seed

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        r, w = prepare(returns, weights, alpha)
        rng = np.random.default_rng(self.seed)
        mu = r.mean(axis=0)
        chol = _cholesky(np.cov(r, rowvar=False, ddof=1))
        sims = mu + rng.standard_normal((self.n_sims, r.shape[1])) @ chol.T
        var, es = empirical_var_es(-(sims @ w), alpha)
        return VarEs(var, es, {"simulations": self.n_sims})
