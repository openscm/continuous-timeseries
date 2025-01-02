"""
Unit tests of `continuous_timeseries.values_at_bounds`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pint
import pytest

from continuous_timeseries.values_at_bounds import ValuesAtBounds

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "values, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0], dtype=np.float32), "m"),
            does_not_raise(),
            id="pint_1d_numpy_float_array",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "m"),
            does_not_raise(),
            id="pint_1d_numpy_int_array",
        ),
        pytest.param(
            Q(1.0, "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional "
                    "but an error was raised while trying to check its shape. "
                    f"Received values={Q(1.0, 'm')}."
                ),
            ),
            id="pint_scalar",
        ),
        pytest.param(
            Q([1.0, 2.0, 3.0], "m"),
            does_not_raise(),
            id="pint_1d_list",
        ),
        pytest.param(
            Q(np.array([[1.0, 2.0], [3.0, 4.0]]), "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional. "
                    "Received `values` with shape (2, 2)"
                ),
            ),
            id="pint_2d_numpy_array",
        ),
        pytest.param(
            Q([[1.0, 2.0], [3.0, 4.0]], "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional. "
                    "Received `values` with shape (2, 2)"
                ),
            ),
            id="pint_2d_list",
        ),
    ),
)
def test_validation(values, expectation):
    with expectation:
        ValuesAtBounds(values)
