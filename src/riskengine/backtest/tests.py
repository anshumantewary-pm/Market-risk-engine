"""Statistical tests for VaR exception series.

All tests take a boolean series of exceptions (``True`` = loss exceeded VaR that day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class TestResult:
    """Likelihood-ratio statistic, its p-value and whether the model is rejected at 5%."""

    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    p_value: float
    dof: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def reject_at_5pct(self) -> bool:
        return self.p_value < 0.05


def _as_bool_array(exceptions) -> np.ndarray:
    x = np.asarray(exceptions, dtype=bool)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("exceptions must be a non-empty 1-D sequence")
    return x


def kupiec_pof(exceptions, alpha: float) -> TestResult:
    """Kupiec proportion-of-failures test: is the exception rate equal to 1 − α?

    ``LR = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x} (x/T)^x]``  ~ χ²(1)

    with p = 1 − α, x exceptions in T days, and 0·ln 0 taken as 0.
    """
    e = _as_bool_array(exceptions)
    t, x = len(e), int(e.sum())
    p = 1 - alpha
    rate = x / t
    log_null = special.xlogy(t - x, 1 - p) + special.xlogy(x, p)
    log_alt = special.xlogy(t - x, 1 - rate) + special.xlogy(x, rate)
    lr = max(0.0, -2 * (log_null - log_alt))
    return TestResult(
        "kupiec_pof", lr, float(stats.chi2.sf(lr, 1)), 1,
        {"observations": t, "exceptions": x, "expected": t * p, "rate": rate},
    )
