import math

import numpy as np
import pandas as pd
import pytest

from riskengine import conventions as cv


def test_to_loss_flips_sign_for_scalars_arrays_and_series():
    assert cv.to_loss(-1.5) == 1.5
    np.testing.assert_array_equal(cv.to_loss(np.array([1.0, -2.0])), [-1.0, 2.0])
    s = pd.Series([0.01, -0.02])
    pd.testing.assert_series_equal(cv.to_loss(s), pd.Series([-0.01, 0.02]))


@pytest.mark.parametrize("alpha", [0.95, 0.99, 0.975])
def test_validate_alpha_accepts_usual_levels(alpha):
    assert cv.validate_alpha(alpha) == alpha


@pytest.mark.parametrize("alpha", [0.5, 1.0, 0.0, 1.2, "0.99", None])
def test_validate_alpha_rejects_bad_levels(alpha):
    with pytest.raises(ValueError):
        cv.validate_alpha(alpha)


def test_check_var_es_accepts_es_at_least_var():
    assert cv.check_var_es(0.02, 0.03) == (0.02, 0.03)
    assert cv.check_var_es(0.02, 0.02) == (0.02, 0.02)


def test_check_var_es_rejects_es_below_var_and_non_finite():
    with pytest.raises(ValueError, match="at least VaR"):
        cv.check_var_es(0.03, 0.02)
    with pytest.raises(ValueError, match="finite"):
        cv.check_var_es(float("nan"), 0.02)


def test_scale_horizon_uses_square_root_of_time():
    assert cv.scale_horizon(1.0, 1) == 1.0
    assert math.isclose(cv.scale_horizon(0.02, 10), 0.02 * math.sqrt(10))


@pytest.mark.parametrize("h", [0, -1, 2.5])
def test_scale_horizon_rejects_bad_horizons(h):
    with pytest.raises(ValueError):
        cv.scale_horizon(0.02, h)


def test_defaults_match_the_design():
    assert cv.DEFAULT_CONFIDENCE_LEVELS == (0.95, 0.99)
    assert cv.DEFAULT_HORIZONS == (1, 10)
    assert cv.DEFAULT_WINDOW == 500
