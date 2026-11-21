"""Synthetic multi-asset prices for demos, tests and CI.

The output is **not real market data**. It is designed to have the features that make
risk modelling interesting: correlated asset classes, calm and stressed volatility regimes,
fat-tailed daily shocks and a few stress episodes. Every number produced from it must be
labelled synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetParams:
    group: str
    annual_vol: float
    annual_drift: float


# Rough, round-number assumptions per ETF proxy. They only shape the synthetic data.
ASSETS: dict[str, AssetParams] = {
    "SPY": AssetParams("us_equity", 0.17, 0.08),
    "QQQ": AssetParams("us_equity", 0.22, 0.10),
    "IWM": AssetParams("us_equity", 0.23, 0.07),
    "EFA": AssetParams("intl_equity", 0.18, 0.05),
    "EEM": AssetParams("intl_equity", 0.24, 0.06),
    "TLT": AssetParams("treasuries", 0.14, 0.03),
    "IEF": AssetParams("treasuries", 0.07, 0.025),
    "LQD": AssetParams("credit", 0.08, 0.035),
    "HYG": AssetParams("credit", 0.10, 0.045),
    "GLD": AssetParams("gold", 0.16, 0.05),
    "USO": AssetParams("oil", 0.35, 0.0),
    "VNQ": AssetParams("real_estate", 0.22, 0.06),
}

GROUPS = ["us_equity", "intl_equity", "treasuries", "credit", "gold", "oil", "real_estate"]

# Correlation between asset classes (upper triangle, row order = GROUPS).
_GROUP_CORR = np.array(
    [
        # useq  intl  tsy   cred  gold  oil   re
        [1.00, 0.80, -0.30, 0.50, 0.00, 0.30, 0.70],
        [0.00, 1.00, -0.25, 0.50, 0.10, 0.35, 0.65],
        [0.00, 0.00, 1.00, 0.30, 0.20, -0.15, -0.10],
        [0.00, 0.00, 0.00, 1.00, 0.10, 0.25, 0.50],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.20, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.25],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ]
)
GROUP_CORR = np.triu(_GROUP_CORR) + np.triu(_GROUP_CORR, 1).T

# Correlation between two different assets in the same class.
WITHIN_GROUP_CORR = {"us_equity": 0.90, "intl_equity": 0.85, "treasuries": 0.90, "credit": 0.60}


def _nearest_correlation(c: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues and rescale to unit diagonal (a simple PSD repair)."""
    vals, vecs = np.linalg.eigh((c + c.T) / 2)
    fixed = vecs @ np.diag(np.clip(vals, floor, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def correlation_matrix(tickers: list[str]) -> np.ndarray:
    """Asset correlation matrix built from class-level and within-class correlations."""
    unknown = [t for t in tickers if t not in ASSETS]
    if unknown:
        raise ValueError(f"no synthetic parameters for: {unknown}")
    gi = [GROUPS.index(ASSETS[t].group) for t in tickers]
    n = len(tickers)
    c = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                c[i, j] = 1.0
            elif gi[i] == gi[j]:
                c[i, j] = WITHIN_GROUP_CORR.get(ASSETS[tickers[i]].group, 0.8)
            else:
                c[i, j] = GROUP_CORR[gi[i], gi[j]]
    return _nearest_correlation(c)


@dataclass(frozen=True)
class StressEpisode:
    """A synthetic sell-off: extra daily drift per asset class, in units of daily volatility."""

    name: str
    start: str
    days: int
    drift_in_vols: dict[str, float]


# Placed on the same dates as the historical windows in config/stress_scenarios.yaml so the
# replay has something to show on sample data. Sizes are invented, not historical.
EPISODES: tuple[StressEpisode, ...] = (
    StressEpisode(
        "synthetic credit crisis", "2008-09-15", 60,
        {"us_equity": -0.35, "intl_equity": -0.40, "credit": -0.30, "treasuries": 0.15,
         "oil": -0.40, "real_estate": -0.45},
    ),
    StressEpisode(
        "synthetic pandemic crash", "2020-02-24", 25,
        {"us_equity": -0.60, "intl_equity": -0.60, "credit": -0.50, "treasuries": 0.20,
         "oil": -0.70, "real_estate": -0.70, "gold": -0.10},
    ),
    StressEpisode(
        "synthetic rate shock", "2022-01-03", 180,
        {"treasuries": -0.12, "credit": -0.10, "us_equity": -0.07, "intl_equity": -0.07,
         "real_estate": -0.09, "oil": 0.08},
    ),
)

P_CALM_TO_STRESSED = 0.01   # expected calm spell ~100 days
P_STRESSED_TO_CALM = 0.05   # expected stressed spell ~20 days
CALM_VOL_MULT = 0.85
STRESSED_VOL_MULT = 2.5
T_DOF = 5                   # Student-t degrees of freedom for daily shocks


def _regimes(n: int, rng: np.random.Generator, forced: np.ndarray) -> np.ndarray:
    """Two-state Markov chain (False = calm, True = stressed); ``forced`` days are stressed."""
    u = rng.random(n)
    state = np.zeros(n, dtype=bool)
    for t in range(1, n):
        p = P_STRESSED_TO_CALM if state[t - 1] else P_CALM_TO_STRESSED
        state[t] = (not state[t - 1]) if u[t] < p else state[t - 1]
        state[t] |= forced[t]
    state[0] = forced[0]
    return state


def generate_prices(
    tickers: list[str] | None = None,
    start: str = "2005-01-03",
    end: str = "2024-12-31",
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Simulate daily adjusted closes on business days between ``start`` and ``end``.

    Daily log return for asset i on day t::

        r = (mu_i - sigma_i^2 / 2) + sigma_i * m_t * z_it + episode drift

    where sigma_i is daily vol, m_t the regime multiplier and z_t a unit-variance
    multivariate Student-t shock (shared chi-square, so assets crash together).
    """
    tickers = list(ASSETS) if tickers is None else list(tickers)
    corr = correlation_matrix(tickers)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, name="Date")
    n, k = len(dates), len(tickers)

    params = [ASSETS[t] for t in tickers]
    sigma = np.array([p.annual_vol for p in params]) / np.sqrt(252)
    mu = np.array([p.annual_drift for p in params]) / 252 - 0.5 * sigma**2

    drift = np.zeros((n, k))
    forced = np.zeros(n, dtype=bool)
    for ep in EPISODES:
        after = dates >= pd.Timestamp(ep.start)
        mask = after & (np.cumsum(after) <= ep.days)
        forced |= mask
        shock = np.array([ep.drift_in_vols.get(p.group, 0.0) for p in params]) * sigma
        drift[mask] += shock

    stressed = _regimes(n, rng, forced)
    mult = np.where(stressed, STRESSED_VOL_MULT, CALM_VOL_MULT)[:, None]

    g = rng.standard_normal((n, k)) @ np.linalg.cholesky(corr).T
    w = rng.chisquare(T_DOF, size=n) / T_DOF
    z = g / np.sqrt(w)[:, None] * np.sqrt((T_DOF - 2) / T_DOF)

    r = mu + sigma * mult * z + drift
    r[0] = 0.0
    prices = start_price * np.exp(np.cumsum(r, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)
