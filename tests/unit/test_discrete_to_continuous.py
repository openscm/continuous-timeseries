"""
Unit tests of `continuous_timeseries.discrete_to_continuous`

(Technically not pure unit tests, but near enough)
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pytest

from continuous_timeseries.discrete_to_continuous.piecewise_constant_next_left_closed import (  # noqa: E501
    PPolyPiecewiseConstantNextLeftClosed,
)
from continuous_timeseries.discrete_to_continuous.piecewise_constant_next_left_open import (  # noqa: E501
    PPolyPiecewiseConstantNextLeftOpen,
)
from continuous_timeseries.discrete_to_continuous.piecewise_constant_previous_left_closed import (  # noqa: E501
    PPolyPiecewiseConstantPreviousLeftClosed,
)
from continuous_timeseries.discrete_to_continuous.piecewise_constant_previous_left_open import (  # noqa: E501
    PPolyPiecewiseConstantPreviousLeftOpen,
)
from continuous_timeseries.exceptions import MissingOptionalDependencyError

piecewise_constant_classes = pytest.mark.parametrize(
    "piecewise_constant_class",
    (
        PPolyPiecewiseConstantNextLeftClosed,
        PPolyPiecewiseConstantNextLeftOpen,
        PPolyPiecewiseConstantPreviousLeftOpen,
        PPolyPiecewiseConstantPreviousLeftClosed,
    ),
)


@pytest.mark.parametrize(
    "x, y, expecation",
    (
        pytest.param(
            np.arange(3),
            np.arange(3),
            does_not_raise(),
            id="valid",
        ),
        pytest.param(
            np.arange(30),
            np.arange(4),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`y` and `self.x` must have the same shape. "
                    "Received: y.shape=(4,). self.x.shape=(30,)"
                ),
            ),
            id="y_shorter_than_x",
        ),
        pytest.param(
            np.arange(3),
            np.arange(4),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`y` and `self.x` must have the same shape. "
                    "Received: y.shape=(4,). self.x.shape=(3,)"
                ),
            ),
            id="y_longer_than_x",
        ),
    ),
)
@piecewise_constant_classes
def test_y_validation(piecewise_constant_class, x, y, expecation):
    with expecation:
        piecewise_constant_class(x=x, y=y)


@pytest.mark.parametrize(
    "x, times, expectation",
    (
        pytest.param(
            np.array([10, 20, 30]),
            np.array([10, 11, 30]),
            does_not_raise(),
            id="no_extrapolation",
        ),
        pytest.param(
            np.array([10, 20, 30]),
            np.array([1.0, 11, 30]),
            pytest.raises(ValueError),
            id="pre_extrapolation",
        ),
        pytest.param(
            np.array([10, 20, 30]),
            np.array([10, 11, 30.5]),
            pytest.raises(ValueError),
            id="post_extrapolation",
        ),
        pytest.param(
            np.array([10, 20, 30]),
            np.array([10, 11, 30.5]),
            pytest.raises(ValueError),
            id="both_extrapolation",
        ),
        pytest.param(
            np.array([-10, 20, 30]),
            np.array([-11, 11, 30.0]),
            pytest.raises(ValueError),
            id="pre_extrapolation_negative_start",
        ),
        pytest.param(
            np.array([-10, -5, -3]),
            np.array([-10, 11, -2.0]),
            pytest.raises(ValueError),
            id="post_extrapolation_all_negative",
        ),
    ),
)
@piecewise_constant_classes
def test_implicit_extrapolation_raises(piecewise_constant_class, x, times, expectation):
    with expectation:
        piecewise_constant_class(x, x)(times)


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="scipy_available"),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=(
                    "`differentiate_piecewise_constant` requires scipy to be installed"
                ),
            ),
            id="scipy_not_available",
        ),
    ),
)
@piecewise_constant_classes
def test_differentiate_scipy_availability(
    piecewise_constant_class, sys_modules_patch, expectation
):
    pytest.importorskip("scipy")
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            piecewise_constant_class(np.arange(10), np.arange(10)).differentiate()


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param(
            {},
            does_not_raise(),
            id="scipy_available",
        ),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=("`integrate_piecewise_constant` requires scipy to be installed"),
            ),
            id="scipy_not_available",
        ),
    ),
)
@piecewise_constant_classes
def test_integrate_scipy_availability(
    piecewise_constant_class, sys_modules_patch, expectation
):
    pytest.importorskip("scipy")
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            piecewise_constant_class(np.arange(10), np.arange(10)).integrate(1.0, 5.0)
