"""Turn prices into returns and portfolio P&L, following ``riskengine.conventions``."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``ln(P_t / P_{t-1})``. The first date is dropped."""
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive to take logs")
    return np.log(prices).diff().iloc[1:]


def align_weights(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Return weights reordered to match the return columns, checking nothing is missing."""
    missing = [t for t in weights.index if t not in asset_returns.columns]
    if missing:
        raise ValueError(f"no returns for weighted tickers: {missing}")
    return weights.astype(float)


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Portfolio return per day, linear in asset returns: ``r_p = sum_i w_i * r_i``."""
    w = align_weights(asset_returns, weights)
    out = asset_returns[w.index] @ w
    out.name = "portfolio_return"
    return out


def portfolio_pnl(asset_returns: pd.DataFrame, weights: pd.Series, notional: float) -> pd.Series:
    """Portfolio P&L in dollars per day: ``notional * r_p``."""
    out = notional * portfolio_returns(asset_returns, weights)
    out.name = "pnl"
    return out
