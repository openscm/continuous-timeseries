"""
Unit tests of `continuous_timeseries.time_axis`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pint
import pint.testing
import pytest

from continuous_timeseries.time_axis import TimeAxis

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "bounds, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0], dtype=np.float32), "yr"),
            does_not_raise(),
            id="pint_1d_numpy_float_array",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "month"),
            does_not_raise(),
            id="pint_1d_numpy_int_array",
        ),
        pytest.param(
            Q(1.0, "yr"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional "
                    "but an error was raised while trying to check its shape. "
                    "Received bounds=1.0 year."
                ),
            ),
            id="pint_scalar",
        ),
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            does_not_raise(),
            id="pint_1d_list",
        ),
        pytest.param(
            Q(np.array([[1.0, 2.0], [3.0, 4.0]]), "hour"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional. "
                    "Received `bounds` with shape (2, 2)"
                ),
            ),
            id="pint_2d_numpy_array",
        ),
        pytest.param(
            Q([[1.0, 2.0], [3.0, 4.0]], "s"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional. "
                    "Received `bounds` with shape (2, 2)"
                ),
            ),
            id="pint_2d_list",
        ),
    ),
)
def test_validation_shape(bounds, expectation):
    with expectation:
        TimeAxis(bounds)


@pytest.mark.parametrize(
    "bounds, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0]), "yr"),
            does_not_raise(),
            id="valid",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "month"),
            does_not_raise(),
            id="valid_int",
        ),
        pytest.param(
            Q(np.array([-1.0, 0.0, 1.0]), "yr"),
            does_not_raise(),
            id="valid_negative_numbers",
        ),
        pytest.param(
            Q(np.array([1.0, 2.0, 1.5]), "yr"),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`bounds` must be strictly monotonically increasing. "
                    "Received bounds=[1.0 2.0 1.5] year"
                ),
            ),
            id="invalid_decreasing",
        ),
        pytest.param(
            Q(np.array([1.0, 2.0, 2.0]), "yr"),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`bounds` must be strictly monotonically increasing. "
                    "Received bounds=[1.0 2.0 2.0] year"
                ),
            ),
            id="invalid_constant",
        ),
    ),
)
def test_validation_monotonically_increasing(bounds, expectation):
    with expectation:
        TimeAxis(bounds)


@pytest.mark.parametrize(
    "bounds, exp",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0]), "yr"),
            Q(np.array([[1.0, 2.0], [2.0, 3.0]]), "yr"),
            id="basic",
        ),
        pytest.param(
            Q(np.array([1.0, 10.0, 30.0]), "yr"),
            Q(np.array([[1.0, 10.0], [10.0, 30.0]]), "yr"),
            id="uneven_spacing",
        ),
    ),
)
def test_bounds2d(bounds, exp):
    res = TimeAxis(bounds).bounds_2d

    pint.testing.assert_equal(res, exp)
