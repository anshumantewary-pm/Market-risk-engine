"""Monte Carlo VaR and ES from simulated one-day asset returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

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


def rank_correlation(r: np.ndarray) -> np.ndarray:
    """Copula correlation from Spearman's rank correlation: ρ = 2·sin(π·ρ_s / 6).

    Rank-based, so it is not distorted by outliers or by the shape of each marginal.
    """
    ranks = np.argsort(np.argsort(r, axis=0), axis=0).astype(float)
    rho_s = np.atleast_2d(np.corrcoef(ranks, rowvar=False))
    rho = 2 * np.sin(np.pi * rho_s / 6)
    np.fill_diagonal(rho, 1.0)
    return rho


def empirical_quantiles(r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Column-wise empirical quantiles of ``r`` at probabilities ``u`` (linear interpolation).

    Equivalent to ``np.quantile(r[:, j], u[:, j])`` per column, but sorts each column once.
    """
    n = len(r)
    grid = np.arange(n, dtype=float)
    sorted_r = np.sort(r, axis=0)
    pos = u * (n - 1)
    return np.column_stack([np.interp(pos[:, j], grid, sorted_r[:, j]) for j in range(r.shape[1])])


class MonteCarloTCopula:
    """Student-t copula for the dependence, each asset's own empirical distribution for marginals.

    The t copula makes extreme moves happen together more often than a normal copula does
    (tail dependence), while the empirical marginals keep each asset's fat tails and skew.
    """

    name = "mc_t_copula"

    def __init__(self, n_sims: int = DEFAULT_SIMULATIONS, seed: int = 42, dof: float = 5.0) -> None:
        if dof <= 2:
            raise ValueError("copula degrees of freedom must be above 2")
        self.n_sims = n_sims
        self.seed = seed
        self.dof = dof

    def estimate(self, returns: pd.DataFrame, weights: pd.Series, alpha: float) -> VarEs:
        r, w = prepare(returns, weights, alpha)
        rng = np.random.default_rng(self.seed)
        chol = _cholesky(rank_correlation(r))
        g = rng.standard_normal((self.n_sims, r.shape[1])) @ chol.T
        chi = rng.chisquare(self.dof, size=self.n_sims) / self.dof
        u = special.stdtr(self.dof, g / np.sqrt(chi)[:, None])
        sims = empirical_quantiles(r, u)
        var, es = empirical_var_es(-(sims @ w), alpha)
        return VarEs(var, es, {"simulations": self.n_sims, "copula_dof": self.dof})
