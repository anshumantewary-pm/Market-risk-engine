import plotly.graph_objects as go
import pytest

from riskengine.dashboard import charts
from riskengine.dashboard.data import OutputsMissing, load_outputs
from riskengine.dashboard.whatif import normalise, recompute, risk_config_from_metadata


@pytest.fixture(scope="module")
def outputs(small_outputs):
    return load_outputs(small_outputs)


def test_load_outputs_reads_every_table(outputs):
    assert outputs.synthetic is True
    assert outputs.notional == 10_000_000
    assert outputs.weights.sum() == pytest.approx(1.0)
    assert {"pnl", "portfolio_return"} <= set(outputs.returns.columns)
    assert outputs.forecasts["exception"].dtype == bool


def test_missing_outputs_tell_the_user_what_to_run(tmp_path):
    with pytest.raises(OutputsMissing, match="make sample"):
        load_outputs(tmp_path)


def test_every_chart_builds(outputs):
    figs = [
        charts.pnl_history(outputs.returns),
        charts.method_comparison(outputs.estimates, 0.99, 1),
        charts.backtest_chart(outputs.forecasts, "historical"),
        charts.attribution_chart(outputs.attribution),
        charts.stress_chart(outputs.stress),
    ]
    assert all(isinstance(f, go.Figure) and f.data for f in figs)


def test_methods_keep_a_fixed_colour(outputs):
    assert len(set(charts.METHOD_COLORS.values())) == len(charts.METHOD_COLORS)
    fig = charts.backtest_chart(outputs.forecasts, "ewma_fhs")
    assert fig.data[1].line.color == charts.METHOD_COLORS["ewma_fhs"]


def test_method_comparison_has_var_and_es_for_six_methods(outputs):
    fig = charts.method_comparison(outputs.estimates, 0.99, 10)
    assert [t.name for t in fig.data] == ["VaR", "Expected Shortfall"]
    assert len(fig.data[0].y) == 6


def test_stress_chart_only_plots_scenarios_that_ran(outputs):
    ran = (outputs.stress["status"] == "run").sum()
    assert len(charts.stress_chart(outputs.stress).data[0].y) == ran


def test_normalise_rescales_and_rejects_all_zero():
    assert normalise({"A": 30, "B": 10}).to_dict() == {"A": 0.75, "B": 0.25}
    assert normalise({"A": -5, "B": 5}).to_dict() == {"A": 0.0, "B": 1.0}
    with pytest.raises(ValueError):
        normalise({"A": 0, "B": 0})


def test_whatif_with_current_weights_matches_pipeline_historical_var(outputs):
    cfg = risk_config_from_metadata(outputs.metadata)
    est, attrib = recompute(outputs.returns, outputs.weights, outputs.notional, cfg)
    ours = est.set_index(["method", "alpha"]).loc[("historical", 0.99), "var_usd"]
    pipe = outputs.estimates.set_index(["method", "alpha", "horizon_days"]).loc[
        ("historical", 0.99, 1), "var_usd"
    ]
    assert ours == pytest.approx(pipe)
    assert attrib["risk_share"].sum() == pytest.approx(1.0)
