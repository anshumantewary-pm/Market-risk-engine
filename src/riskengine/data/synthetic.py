"""Synthetic multi-asset prices for demos, tests and CI.

The output is **not real market data**. It is designed to have the features that make
risk modelling interesting: correlated asset classes, calm and stressed volatility regimes,
fat-tailed daily shocks and a few stress episodes. Every number produced from it must be
labelled synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
