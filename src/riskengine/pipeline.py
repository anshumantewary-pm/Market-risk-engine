"""End-to-end risk run.

Usage::

    python -m riskengine.pipeline            # real data in data/raw/prices.csv
    python -m riskengine.pipeline --sample   # generate synthetic prices first

Steps: load config -> prices -> returns -> headline VaR/ES -> rolling backtest ->
attribution -> stress -> write ``data/processed/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from riskengine.config import DEFAULT_CONFIG_PATH, DEFAULT_STRESS_PATH
from riskengine.data.load import DEFAULT_PRICES_PATH

DEFAULT_OUT_DIR = Path("data/processed")


@dataclass(frozen=True)
class RunSettings:
    config_path: Path = DEFAULT_CONFIG_PATH
    stress_path: Path = DEFAULT_STRESS_PATH
    prices_path: Path = DEFAULT_PRICES_PATH
    out_dir: Path = DEFAULT_OUT_DIR
    sample: bool = False
    sample_start: str = "2005-01-03"
    sample_end: str = "2024-12-31"
    backtest_days: int | None = None   # None = every day after the first window
    backtest_sims: int = 5_000         # Monte Carlo simulations per backtest forecast
    verbose: bool = True


def parse_args(argv: list[str] | None = None) -> RunSettings:
    p = argparse.ArgumentParser(description="Run the market risk pipeline.")
    p.add_argument("--sample", action="store_true",
                   help="generate SYNTHETIC prices instead of using downloaded data")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--stress-config", default=str(DEFAULT_STRESS_PATH))
    p.add_argument("--prices", default=str(DEFAULT_PRICES_PATH))
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sample-start", default=RunSettings.sample_start)
    p.add_argument("--sample-end", default=RunSettings.sample_end)
    p.add_argument("--backtest-days", type=int, default=None,
                   help="only backtest the most recent N days (default: all)")
    p.add_argument("--backtest-sims", type=int, default=RunSettings.backtest_sims)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    return RunSettings(
        config_path=Path(a.config), stress_path=Path(a.stress_config),
        prices_path=Path(a.prices), out_dir=Path(a.out), sample=a.sample,
        sample_start=a.sample_start, sample_end=a.sample_end,
        backtest_days=a.backtest_days, backtest_sims=a.backtest_sims, verbose=not a.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    print(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
