"""
Integration tests of our piecewise constant discrete to continuous conversion and back.

Implicitly, tests of `continuous_timeseries.discrete_to_continuous`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise
from typing import Union

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
)
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR
from continuous_timeseries.warnings import (
    InterpolationUpdateChangedValuesAtBoundsWarning,
)

UR = pint.get_application_registry()
Q = UR.Quantity


@define
class PiecewiseConstantTestCase:
    """
    Test case for piecewise constant interpolation
    """

    name: str
    interpolation: InterpolationOption
    x: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    y: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    exp_extrapolate_pre: PINT_SCALAR
    exp_first_edge: PINT_SCALAR
    exp_first_window: PINT_SCALAR
    exp_internal_edge: PINT_SCALAR
    exp_last_window: PINT_SCALAR
    exp_last_edge: PINT_SCALAR
    exp_extrapolate_post: PINT_SCALAR
    exp_round_trip_values_at_bounds_same: bool
    expectation_to_continuous_timeseries: Union[
        does_not_raise, pytest.RaisesContext
    ] = field()
    ts: Timeseries = field()

    @expectation_to_continuous_timeseries.default
    def initialise_expectation_to_continuous_timeseries(self):
        if self.exp_round_trip_values_at_bounds_same:
            return does_not_raise()
        else:
            return pytest.warns(
                InterpolationUpdateChangedValuesAtBoundsWarning,
                match=re.escape(
                    f"Using the interpolation {self.interpolation.name} means "
                    "that the y-values do not line up exactly with the x-values. "
                    "In other words, in the output, "
                    "y(x(1)) is not equal to the input y(1), "
                    "y(x(2)) is not equal to the input y(2), "
                    "y(x(n)) is not equal to the input y(n) etc. "
                    "This may cause confusion. "
                    "Either ignore this warning, "
                    "suppress it "
                    "(by passing `warn_if_values_at_bounds_could_confuse=False` "
                    "or via Python's `warnings` module settings) "
                    "or choose a different interpolation option."
                ),
            )

    @ts.default
    def initialise_timeseries(self):
        return Timeseries.from_arrays(
            x=self.x,
            y=self.y,
            interpolation=self.interpolation,
            name=self.name,
        )


piecewise_constant_test_cases = pytest.mark.parametrize(
    "piecewise_constant_test_case",
    (
        pytest.param(
            PiecewiseConstantTestCase(
                name="piecewise_constant_next_left_closed",
                interpolation=InterpolationOption.PiecewiseConstantNextLeftClosed,
                x=Q([1750, 1850, 2000], "yr"),
                y=Q([0.0, 2.0, 4.0], "W"),
                exp_extrapolate_pre=Q(0.0, "W"),
                exp_first_edge=Q(2.0, "W"),
                exp_first_window=Q(2.0, "W"),
                exp_internal_edge=Q(4.0, "W"),
                exp_last_window=Q(4.0, "W"),
                exp_last_edge=Q(4.0, "W"),
                exp_extrapolate_post=Q(4.0, "W"),
                exp_round_trip_values_at_bounds_same=False,
            ),
            id="piecewise_constant_next_left_closed",
        ),
        pytest.param(
            PiecewiseConstantTestCase(
                name="piecewise_constant_next_left_open",
                interpolation=InterpolationOption.PiecewiseConstantNextLeftOpen,
                x=Q([1750, 1850, 2000], "yr"),
                y=Q([0.0, 2.0, 4.0], "W"),
                exp_extrapolate_pre=Q(0.0, "W"),
                exp_first_edge=Q(0.0, "W"),
                exp_first_window=Q(2.0, "W"),
                exp_internal_edge=Q(2.0, "W"),
                exp_last_window=Q(4.0, "W"),
                exp_last_edge=Q(4.0, "W"),
                exp_extrapolate_post=Q(4.0, "W"),
                exp_round_trip_values_at_bounds_same=True,
            ),
            id="piecewise_constant_next_left_open",
        ),
        pytest.param(
            PiecewiseConstantTestCase(
                name="piecewise_constant_previous_left_closed",
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
                x=Q([1750, 1850, 2000], "yr"),
                y=Q([0.0, 2.0, 4.0], "W"),
                exp_extrapolate_pre=Q(0.0, "W"),
                exp_first_edge=Q(0.0, "W"),
                exp_first_window=Q(0.0, "W"),
                exp_internal_edge=Q(2.0, "W"),
                exp_last_window=Q(2.0, "W"),
                exp_last_edge=Q(4.0, "W"),
                exp_extrapolate_post=Q(4.0, "W"),
                exp_round_trip_values_at_bounds_same=True,
            ),
            id="piecewise_constant_previous_left_closed",
        ),
        pytest.param(
            PiecewiseConstantTestCase(
                name="piecewise_constant_previous_left_open",
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftOpen,
                x=Q([1750, 1850, 2000], "yr"),
                y=Q([0.0, 2.0, 4.0], "W"),
                exp_extrapolate_pre=Q(0.0, "W"),
                exp_first_edge=Q(0.0, "W"),
                exp_first_window=Q(0.0, "W"),
                exp_internal_edge=Q(0.0, "W"),
                exp_last_window=Q(2.0, "W"),
                exp_last_edge=Q(2.0, "W"),
                exp_extrapolate_post=Q(4.0, "W"),
                exp_round_trip_values_at_bounds_same=False,
            ),
            id="piecewise_constant_previous_left_open",
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
        piecewise_constant_test_case.x,
    )


@piecewise_constant_test_cases
def test_implicit_extrapolation_pre_raises(piecewise_constant_test_case):
    pre_domain_time = piecewise_constant_test_case.x[0] - Q(1, "yr")

    with pytest.raises(ExtrapolationNotAllowedError):
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            pre_domain_time
        )


@piecewise_constant_test_cases
def test_extrapolation_pre(piecewise_constant_test_case):
    pre_domain_time = piecewise_constant_test_case.x[0] - Q(1, "yr")

    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            pre_domain_time,
            allow_extrapolation=True,
        ),
        piecewise_constant_test_case.exp_extrapolate_pre,
    )


@piecewise_constant_test_cases
def test_first_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.x[0],
        ),
        piecewise_constant_test_case.exp_first_edge,
    )


@piecewise_constant_test_cases
def test_first_window_value(piecewise_constant_test_case):
    first_window_time = (
        piecewise_constant_test_case.x[0] + piecewise_constant_test_case.x[1]
    ) / 2.0
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            first_window_time,
        ),
        piecewise_constant_test_case.exp_first_window,
    )


@piecewise_constant_test_cases
def test_internal_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.x[1],
        ),
        piecewise_constant_test_case.exp_internal_edge,
    )


@piecewise_constant_test_cases
def test_last_window_value(piecewise_constant_test_case):
    last_window_time = (
        piecewise_constant_test_case.x[-1] + piecewise_constant_test_case.x[-2]
    ) / 2.0
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            last_window_time,
        ),
        piecewise_constant_test_case.exp_last_window,
    )


@piecewise_constant_test_cases
def test_last_edge_value(piecewise_constant_test_case):
    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            piecewise_constant_test_case.x[-1],
        ),
        piecewise_constant_test_case.exp_last_edge,
    )


@piecewise_constant_test_cases
def test_implicit_extrapolation_post_raises(piecewise_constant_test_case):
    post_domain_time = piecewise_constant_test_case.x[-1] + Q(1, "yr")

    with pytest.raises(ExtrapolationNotAllowedError):
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            post_domain_time
        )


@piecewise_constant_test_cases
def test_extrapolation_post(piecewise_constant_test_case):
    post_domain_time = piecewise_constant_test_case.x[-1] + Q(1, "yr")

    pint.testing.assert_equal(
        piecewise_constant_test_case.ts.timeseries_continuous.interpolate(
            post_domain_time,
            allow_extrapolation=True,
        ),
        piecewise_constant_test_case.exp_extrapolate_post,
    )


@piecewise_constant_test_cases
def test_discrete_to_continuous_equivalence(piecewise_constant_test_case):
    res = discrete_to_continuous(
        x=piecewise_constant_test_case.x,
        y=piecewise_constant_test_case.y,
        name=piecewise_constant_test_case.name,
        interpolation=piecewise_constant_test_case.interpolation,
    )

    exp = piecewise_constant_test_case.ts.timeseries_continuous

    assert res.name == exp.name
    assert res.time_units == exp.time_units
    assert res.values_units == exp.values_units
    for res_v, exp_v in zip(res.domain, exp.domain):
        pint.testing.assert_equal(res_v, exp_v)

    check_times = (
        np.linspace(
            piecewise_constant_test_case.x[0].m,
            piecewise_constant_test_case.x[-1].m,
            100,
        )
        * piecewise_constant_test_case.x.u
    )
    pint.testing.assert_equal(
        res.interpolate(check_times), exp.interpolate(check_times)
    )


@piecewise_constant_test_cases
def test_round_tripping(piecewise_constant_test_case):
    start = TimeseriesDiscrete(
        name=piecewise_constant_test_case.name,
        time_axis=TimeAxis(piecewise_constant_test_case.x),
        values_at_bounds=ValuesAtBounds(piecewise_constant_test_case.y),
    )

    with piecewise_constant_test_case.expectation_to_continuous_timeseries:
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
        np.hstack(
            [
                piecewise_constant_test_case.exp_first_edge,
                piecewise_constant_test_case.exp_internal_edge,
                piecewise_constant_test_case.exp_last_edge,
            ]
        ),
    )

    if piecewise_constant_test_case.exp_round_trip_values_at_bounds_same:
        # This holds true only in select cases
        pint.testing.assert_equal(
            res.values_at_bounds.values,
            start.values_at_bounds.values,
        )
