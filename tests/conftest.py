from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def small_outputs(tmp_path_factory) -> Path:
    """Run the pipeline once on a short synthetic history; return the output directory."""
    from riskengine.pipeline import RunSettings, run

    tmp = tmp_path_factory.mktemp("dashboard_run")
    raw = yaml.safe_load((ROOT / "config" / "portfolio.yaml").read_text())
    raw["risk"].update(window=120, mc_simulations=2000)
    cfg = tmp / "portfolio.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    run(RunSettings(
        config_path=cfg, stress_path=ROOT / "config" / "stress_scenarios.yaml",
        prices_path=tmp / "prices.csv", out_dir=tmp / "processed", sample=True,
        sample_start="2019-01-01", sample_end="2021-06-30", backtest_days=40,
        backtest_sims=1000, verbose=False,
    ))
    return tmp / "processed"
