"""Turn prices into returns and portfolio P&L, following ``riskengine.conventions``."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``ln(P_t / P_{t-1})``. The first date is dropped."""
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive to take logs")
    return np.log(prices).diff().iloc[1:]
