"""
Integration tests of our linear/higher-order discrete to continuous conversion and back.

Implicitly, tests of `continuous_timeseries.discrete_to_continuous`
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest
from attrs import define, field, validators

from continuous_timeseries import (
    InterpolationOption,
    TimeAxis,
    Timeseries,
    TimeseriesDiscrete,
    ValuesAtBounds,
)
from continuous_timeseries.discrete_to_continuous import discrete_to_continuous
from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR

UR = pint.get_application_registry()
Q = UR.Quantity


@define
class LinearAndHigherOrderTestCase:
    """
    Test case for linear and higher-order interpolation
    """

    name: str
    interpolation: InterpolationOption
    time_axis_bounds: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    values_at_bounds: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    exp_extrapolate_one_year_pre: PINT_SCALAR
    exp_middle_first_window: PINT_SCALAR
    exp_middle_last_window: PINT_SCALAR
    exp_extrapolate_one_year_post: PINT_SCALAR
    ts: Timeseries = field()

    @ts.default
    def initialise_timeseries(self):
        return Timeseries.from_arrays(
            time_axis_bounds=self.time_axis_bounds,
            values_at_bounds=self.values_at_bounds,
            interpolation=self.interpolation,
            name=self.name,
        )


piecewise_constant_test_cases = pytest.mark.parametrize(
    "piecewise_constant_test_case",
    (
        pytest.param(
            LinearAndHigherOrderTestCase(
                name="linear",
                interpolation=InterpolationOption.Linear,
                time_axis_bounds=Q([1750, 1850, 2000], "yr"),
                values_at_bounds=Q([0.0, 2.0, 3.0], "W"),
                exp_extrapolate_one_year_pre=Q(-1 / 50.0, "W"),
                exp_middle_first_window=Q(1.0, "W"),
                exp_middle_last_window=Q(2.5, "W"),
                exp_extrapolate_one_year_post=Q(3.0 + 1.0 / 150.0, "W"),
            ),
            id="linear",
        ),
    ),
)


@piecewise_constant_test_cases
def test_name_set_correctly(piecewise_constant_test_case):
    assert piecewise_constant_test_case.ts.name == piecewise_constant_test_case.name


@piecewise_constant_test_cases
def test_time_axis_set_correctly(piecewise_constant_test_case):
    assert isinstance(piecewise_constant_test_case.ts.time_axis, TimeAxis)

    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.time_axis.bounds,
        piecewise_constant_test_case.time_axis_bounds,
    )


@piecewise_constant_test_cases
def test_implicit_extrapolation_pre_raises(piecewise_constant_test_case):
    pre_domain_time = piecewise_constant_test_case.time_axis_bounds[0] - Q(1, "yr")

    with pytest.raises(ExtrapolationNotAllowedError):
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            pre_domain_time
        )


@piecewise_constant_test_cases
def test_extrapolation_pre(piecewise_constant_test_case):
    pre_domain_time = piecewise_constant_test_case.time_axis_bounds[0] - Q(1, "yr")

    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            pre_domain_time,
            allow_extrapolation=True,
        ),
        piecewise_constant_test_case.exp_extrapolate_one_year_pre,
    )


@piecewise_constant_test_cases
def test_first_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.time_axis_bounds[0],
        ),
        piecewise_constant_test_case.values_at_bounds[0],
    )


@piecewise_constant_test_cases
def test_first_window_value(piecewise_constant_test_case):
    first_window_time = (
        piecewise_constant_test_case.time_axis_bounds[0]
        + piecewise_constant_test_case.time_axis_bounds[1]
    ) / 2.0
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            first_window_time,
        ),
        piecewise_constant_test_case.exp_middle_first_window,
    )


@piecewise_constant_test_cases
def test_internal_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.time_axis_bounds[1],
        ),
        piecewise_constant_test_case.values_at_bounds[1],
    )


@piecewise_constant_test_cases
def test_last_window_value(piecewise_constant_test_case):
    last_window_time = (
        piecewise_constant_test_case.time_axis_bounds[-1]
        + piecewise_constant_test_case.time_axis_bounds[-2]
    ) / 2.0
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            last_window_time,
        ),
        piecewise_constant_test_case.exp_middle_last_window,
    )


@piecewise_constant_test_cases
def test_last_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.time_axis_bounds[-1],
        ),
        piecewise_constant_test_case.values_at_bounds[-1],
    )


@piecewise_constant_test_cases
def test_implicit_extrapolation_post_raises(piecewise_constant_test_case):
    post_domain_time = piecewise_constant_test_case.time_axis_bounds[-1] + Q(1, "yr")

    with pytest.raises(ExtrapolationNotAllowedError):
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            post_domain_time
        )


@piecewise_constant_test_cases
def test_extrapolation_post(piecewise_constant_test_case):
    post_domain_time = piecewise_constant_test_case.time_axis_bounds[-1] + Q(1, "yr")

    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            post_domain_time,
            allow_extrapolation=True,
        ),
        piecewise_constant_test_case.exp_extrapolate_one_year_post,
    )


@piecewise_constant_test_cases
def test_discrete_to_continuous_equivalence(piecewise_constant_test_case):
    ts_discrete = TimeseriesDiscrete(
        name=piecewise_constant_test_case.name,
        time_axis=TimeAxis(piecewise_constant_test_case.time_axis_bounds),
        values_at_bounds=ValuesAtBounds(piecewise_constant_test_case.values_at_bounds),
    )

    res = discrete_to_continuous(
        ts_discrete, interpolation=piecewise_constant_test_case.interpolation
    )

    exp = piecewise_constant_test_case.ts.timeseries_continuous

    assert res.name == exp.name
    assert res.time_units == exp.time_units
    assert res.values_units == exp.values_units
    for res_v, exp_v in zip(res.domain, exp.domain):
        pint.testing.assert_equal(res_v, exp_v)

    check_times = (
        np.linspace(
            piecewise_constant_test_case.time_axis_bounds[0].m,
            piecewise_constant_test_case.time_axis_bounds[-1].m,
            100,
        )
        * piecewise_constant_test_case.time_axis_bounds.u
    )
    pint.testing.assert_equal(
        res.interpolate(check_times), exp.interpolate(check_times)
    )


@piecewise_constant_test_cases
def test_round_tripping(piecewise_constant_test_case):
    start = TimeseriesDiscrete(
        name=piecewise_constant_test_case.name,
        time_axis=TimeAxis(piecewise_constant_test_case.time_axis_bounds),
        values_at_bounds=ValuesAtBounds(piecewise_constant_test_case.values_at_bounds),
    )

    continuous = start.to_continuous_timeseries(
        piecewise_constant_test_case.interpolation
    )

    res = continuous.to_discrete_timeseries(start.time_axis)

    assert res.name == start.name
    pint.testing.assert_equal(
        res.time_axis.bounds,
        start.time_axis.bounds,
    )

    # This holds true in all cases
    pint.testing.assert_equal(
        res.values_at_bounds.values,
        start.values_at_bounds.values,
    )


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="scipy_available"),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=(
                    "`discrete_to_continuous_linear` requires scipy to be installed"
                ),
            ),
            id="scipy_not_available",
        ),
    ),
)
def test_scipy_missing_error_discrete_to_continuous_linear(
    sys_modules_patch, expectation
):
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            Timeseries.from_arrays(
                time_axis_bounds=Q([1850, 1900], "yr"),
                values_at_bounds=Q([1, 2], "kg"),
                interpolation=InterpolationOption.Linear,
                name="test",
            )


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="scipy_available"),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=(
                    "`discrete_to_continuous_higher_order` "
                    "requires scipy to be installed"
                ),
            ),
            id="scipy_not_available",
        ),
    ),
)
@pytest.mark.parametrize(
    "interpolation",
    (
        InterpolationOption.Quadratic,
        InterpolationOption.Cubic,
        InterpolationOption.Quartic,
    ),
)
def test_scipy_missing_error_discrete_to_continuous_higher_order(
    interpolation, sys_modules_patch, expectation
):
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            Timeseries.from_arrays(
                time_axis_bounds=Q(np.arange(100), "yr"),
                values_at_bounds=Q(np.arange(100), "kg"),
                interpolation=interpolation,
                name="test",
            )
