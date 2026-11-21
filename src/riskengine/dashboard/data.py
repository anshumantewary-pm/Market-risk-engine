"""Load the pipeline's outputs for the dashboard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from riskengine.pipeline import DEFAULT_OUT_DIR, OUTPUT_FILES

OUTPUT_DIR_ENV = "RISKENGINE_OUTPUT_DIR"


class OutputsMissing(FileNotFoundError):
    """Raised when the pipeline has not been run yet."""


@dataclass(frozen=True)
class Outputs:
    returns: pd.DataFrame
    estimates: pd.DataFrame
    forecasts: pd.DataFrame
    backtest_summary: pd.DataFrame
    attribution: pd.DataFrame
    stress: pd.DataFrame
    metadata: dict

    @property
    def synthetic(self) -> bool:
        return bool(self.metadata.get("synthetic", True))

    @property
    def weights(self) -> pd.Series:
        return pd.Series(self.metadata["portfolio"]["weights"], dtype=float)

    @property
    def notional(self) -> float:
        return float(self.metadata["portfolio"]["notional"])


def output_dir() -> Path:
    """Where to read outputs from: ``$RISKENGINE_OUTPUT_DIR`` or ``data/processed``."""
    return Path(os.environ.get(OUTPUT_DIR_ENV, DEFAULT_OUT_DIR))


def load_outputs(directory: str | Path | None = None) -> Outputs:
    d = Path(directory) if directory is not None else output_dir()
    missing = [n for n in (*OUTPUT_FILES, "run_metadata.json") if not (d / n).exists()]
    if missing:
        raise OutputsMissing(
            f"pipeline outputs not found in {d} (missing: {', '.join(missing)}). "
            "Run `make sample` (synthetic data) or `make pipeline` (real data) first."
        )
    return Outputs(
        returns=pd.read_csv(d / "returns.csv", index_col=0, parse_dates=True),
        estimates=pd.read_csv(d / "estimates.csv", parse_dates=["window_start", "window_end"]),
        forecasts=pd.read_csv(d / "backtest_forecasts.csv", parse_dates=["date"]),
        backtest_summary=pd.read_csv(d / "backtest_summary.csv", index_col="method"),
        attribution=pd.read_csv(d / "attribution.csv", index_col="ticker"),
        stress=pd.read_csv(d / "stress.csv"),
        metadata=json.loads((d / "run_metadata.json").read_text()),
    )
