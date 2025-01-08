"""
Integration tests of `continuous_timeseries.timeseries`
"""

from __future__ import annotations

import itertools
import re
import sys
from contextlib import nullcontext as does_not_raise
from functools import partial
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest
from attrs import define, field, validators

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries import (
    Timeseries,
    UnreachableIntegralPreservingInterpolationTarget,
)
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR
from continuous_timeseries.values_at_bounds import ValuesAtBounds
from continuous_timeseries.warnings import (
    InterpolationUpdateChangedValuesAtBoundsWarning,
)

# Tests don't make sense without scipy
pytest.importorskip("scipy")

UR = pint.get_application_registry()
Q = UR.Quantity


formatting_check_cases = pytest.mark.parametrize(
    "ts",
    (
        pytest.param(
            Timeseries.from_arrays(
                x=Q([1.0, 10.0, 20.0], "yr"),
                y=Q([10.0, 12.0, 32.0], "Mt / yr"),
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
                name="piecewise_constant",
            ),
            id="piecewise_constant",
        ),
        pytest.param(
            Timeseries.from_arrays(
                x=Q([1.0, 10.0, 20.0], "yr"),
                y=Q([10.0, 12.0, 32.0], "Mt / yr"),
                interpolation=InterpolationOption.Linear,
                name="piecewise_linear",
            ),
            id="piecewise_linear",
        ),
        pytest.param(
            Timeseries.from_arrays(
                x=Q(np.arange(1750.0, 3000.0 + 1), "yr"),
                y=Q(10.0 + np.arange(1251.0), "Mt / yr"),
                interpolation=InterpolationOption.Linear,
                name="piecewise_linear_heaps_of_windows",
            ),
            id="piecewise_linear_heaps_of_windows",
        ),
    ),
)


@formatting_check_cases
def test_repr(ts, file_regression):
    exp = (
        "Timeseries("
        f"time_axis={ts.time_axis!r}, "
        f"timeseries_continuous={ts.timeseries_continuous!r}"
        ")"
    )

    assert repr(ts) == exp

    # Avoid changing addresses causing issues
    file_regression_value = re.sub("at .*>", "at address>", repr(ts))

    file_regression.check(
        f"{file_regression_value}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_str(ts, file_regression):
    exp = (
        "Timeseries("
        f"time_axis={ts.time_axis}, "
        f"timeseries_continuous={ts.timeseries_continuous}"
        ")"
    )

    assert str(ts) == exp

    file_regression.check(
        f"{ts}\n",
        extension=".txt",
    )


@pytest.mark.xfail(
    condition=not (sys.version_info >= (3, 10)),
    reason="shape info only in Python>=3.10",
)
@formatting_check_cases
def test_pretty(ts, file_regression):
    pytest.importorskip("IPython")

    from IPython.lib.pretty import pretty

    file_regression.check(
        f"{pretty(ts)}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_html(ts, file_regression):
    file_regression.check(
        f"{ts._repr_html_()}\n",
        extension=".html",
    )


@define
class OperationsTestCase:
    """A test case for operations with `Timeseries`"""

    name: str
    interpolation: InterpolationOption
    x: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    y: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )

    time_interp: PINT_NUMPY_ARRAY
    """Times to use for checking interpolation"""

    exp_interp: PINT_NUMPY_ARRAY
    """Expected values of interpolation at `time_interp`"""

    time_extrap: PINT_NUMPY_ARRAY
    """Times to use for checking extrapolation"""

    exp_extrap: PINT_NUMPY_ARRAY
    """Expected values of extrapolation at `time_extrap`"""

    time_derivative: PINT_NUMPY_ARRAY
    """Times to use for checking differentiation"""

    exp_derivative: PINT_NUMPY_ARRAY
    """Expected values of the derivative at `time_derivative`"""

    time_integral: PINT_NUMPY_ARRAY
    """Times to use for checking integration"""

    integration_constant_integral: PINT_SCALAR
    """Integration constant to use for checking integration"""

    exp_integral: PINT_NUMPY_ARRAY
    """Expected values of the integral at `time_integral`"""

    ts: Timeseries = field()

    @ts.default
    def initialise_timeseries(self):
        return Timeseries.from_arrays(
            x=self.x,
            y=self.y,
            interpolation=self.interpolation,
            name=self.name,
        )


operations_test_cases = pytest.mark.parametrize(
    "operations_test_case",
    (
        pytest.param(
            OperationsTestCase(
                name="piecewise_constant_next_left_closed",
                interpolation=InterpolationOption.PiecewiseConstantNextLeftClosed,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([-1.0, 0.0, 2.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([0.0, 0.0, 2.0, 2.0, 2.0], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([-1.0, 2.0, 2.0], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(10.0, "Gt yr"),
                exp_integral=(
                    Q(
                        np.cumsum(
                            [
                                10.0,  # integration constant
                                # y = c
                                # int y dx = c * dx + const
                                -1.0 * 5.0,
                                0.0,
                                0.0,
                                2.0 * 10.0,
                                2.0 * 20.0,
                                2.0 * 10.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="piecewise_constant_next_left_closed",
        ),
        pytest.param(
            OperationsTestCase(
                name="piecewise_constant_next_left_open",
                interpolation=InterpolationOption.PiecewiseConstantNextLeftOpen,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([-1.0, 0.0, 2.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([-1.0, 0.0, 0.0, 2.0, 2.0], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([-1.0, 0.0, 2.0], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(10.0, "Gt yr"),
                exp_integral=(
                    Q(
                        np.cumsum(
                            [
                                10.0,  # integration constant
                                # y = c
                                # int y dx = c * dx + const
                                -1.0 * 5.0,
                                0.0,
                                0.0,
                                2.0 * 10.0,
                                2.0 * 20.0,
                                2.0 * 10.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="piecewise_constant_next_left_open",
        ),
        pytest.param(
            OperationsTestCase(
                name="piecewise_constant_previous_left_closed",
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([-1.0, 0.0, 2.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([-1.0, -1.0, 0.0, 0.0, 2.0], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([-1.0, 0.0, 2.0], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(5.2, "Gt yr"),
                exp_integral=(
                    Q(
                        np.cumsum(
                            [
                                5.2,  # integration constant
                                # y = c
                                # int y dx = c * dx + const
                                -1.0 * 5.0,
                                -1.0 * 5.0,
                                -1.0 * 5.0,
                                0.0 * 10.0,
                                0.0 * 20.0,
                                2.0 * 10.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="piecewise_constant_previous_left_closed",
        ),
        pytest.param(
            OperationsTestCase(
                name="piecewise_constant_previous_left_open",
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftOpen,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([-1.0, 0.0, 2.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([-1.0, -1.0, -1.0, 0.0, 0.0], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([-1.0, -1.0, 2.0], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(5.2, "Gt yr"),
                exp_integral=(
                    Q(
                        np.cumsum(
                            [
                                5.2,  # integration constant
                                # y = c
                                # int y dx = c * dx + const
                                -1.0 * 5.0,
                                -1.0 * 5.0,
                                -1.0 * 5.0,
                                0.0 * 10.0,
                                0.0 * 20.0,
                                2.0 * 10.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="piecewise_constant_previous_left_open",
        ),
        pytest.param(
            OperationsTestCase(
                name="linear",
                interpolation=InterpolationOption.Linear,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([-1.0, 0.0, 2.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([-1.0, -0.5, 0.0, 2.0 / 3.0, 2.0], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([-1.5, 0.0, 2.0 + 2.0 / 3.0], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [
                        1.0 / 10.0,
                        1.0 / 10.0,
                        1.0 / 10.0,
                        # On boundary, get the value from the next window
                        # (next closed logic).
                        2.0 / 30.0,
                        2.0 / 30.0,
                        2.0 / 30.0,
                        2.0 / 30.0,
                    ],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(10.0, "Gt yr"),
                exp_integral=(
                    Q(10.0, "Gt yr")
                    + Q(
                        np.cumsum(
                            [
                                0.0,
                                # y = mx + c
                                # int y dx = m dx^2 / 2 + c dx + const
                                0.1 * 5.0**2 / 2 - 1.5 * 5.0,
                                0.1 * 5.0**2 / 2 - 1.0 * 5.0,
                                0.1 * 5.0**2 / 2 - 0.5 * 5.0,
                                2.0 / 30.0 * 10.0**2 / 2 + 0.0 * 10.0,
                                2.0 / 30.0 * 20.0**2 / 2 + 2.0 / 3.0 * 20.0,
                                2.0 / 30.0 * 10.0**2 / 2 + 2.0 * 10.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="linear",
        ),
        pytest.param(
            OperationsTestCase(
                name="quadratic",
                interpolation=InterpolationOption.Quadratic,
                x=Q([2010, 2020, 2050], "yr"),
                y=Q([0.0, 1.0, 16.0], "Gt"),
                time_interp=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_interp=Q([0.0, 0.5**2, 1.0, 2.0**2, 4.0**2], "Gt"),
                time_extrap=Q([2005.0, 2020.0, 2060.0], "yr"),
                exp_extrap=Q([0.5**2, 1.0, 5.0**2], "Gt"),
                time_derivative=Q(
                    [2000.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                exp_derivative=Q(
                    [
                        # y = ((x - 2010) / 10)^2
                        # dy/dx = 2(x - 2010) / 10 * 1 / 10 = (x - 2010) / 50
                        -10.0 / 50.0,
                        0.0,
                        5.0 / 50.0,
                        10.0 / 50.0,
                        20.0 / 50.0,
                        40.0 / 50.0,
                        50.0 / 50.0,
                    ],
                    "Gt / yr",
                ),
                time_integral=Q(
                    [2005.0, 2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2060.0], "yr"
                ),
                integration_constant_integral=Q(-1.0, "Gt yr"),
                exp_integral=(
                    Q(-1.0, "Gt yr")
                    + Q(
                        np.cumsum(
                            [
                                0.0,
                                # y = ((x - 2010) / 10)^2
                                # int y dx = ((x - 2010) / 10)^3 / 3.0 * 10 + const
                                ((0.0 / 10) ** 3 - (-5.0 / 10.0) ** 3) * 10 / 3.0,
                                ((5.0 / 10) ** 3 - (0.0 / 10.0) ** 3) * 10 / 3.0,
                                ((10.0 / 10) ** 3 - (5.0 / 10.0) ** 3) * 10 / 3.0,
                                ((20.0 / 10) ** 3 - (10.0 / 10.0) ** 3) * 10 / 3.0,
                                ((40.0 / 10) ** 3 - (20.0 / 10.0) ** 3) * 10 / 3.0,
                                ((50.0 / 10) ** 3 - (40.0 / 10.0) ** 3) * 10 / 3.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
            ),
            id="quadratic",
        ),
    ),
)


@operations_test_cases
def test_discrete(operations_test_case):
    exp = TimeseriesDiscrete(
        name=operations_test_case.name,
        time_axis=TimeAxis(operations_test_case.x),
        # Discrete is interpolated values,
        # which is not always what went in i.e. `operations_test_case.y`.
        values_at_bounds=ValuesAtBounds(
            operations_test_case.ts.timeseries_continuous.interpolate(
                operations_test_case.x
            )
        ),
    )

    res = operations_test_case.ts.discrete

    assert res.name == exp.name
    pint.testing.assert_equal(res.time_axis.bounds, exp.time_axis.bounds)
    pint.testing.assert_equal(
        res.values_at_bounds.values,
        exp.values_at_bounds.values,
    )


@operations_test_cases
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_differentiate(operations_test_case, name_res):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    derivative = operations_test_case.ts.interpolate(
        operations_test_case.time_derivative, allow_extrapolation=True
    ).differentiate(**kwargs)

    if name_res is None:
        assert derivative.name == f"{operations_test_case.ts.name}_derivative"
    else:
        assert derivative.name == name_res

    assert isinstance(derivative, Timeseries)

    pint.testing.assert_equal(
        operations_test_case.time_derivative, derivative.time_axis.bounds
    )

    pint.testing.assert_allclose(
        derivative.interpolate(
            time_axis=operations_test_case.time_derivative
        ).discrete.values_at_bounds.values,
        operations_test_case.exp_derivative,
        rtol=1e-10,
    )


@operations_test_cases
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_integrate(operations_test_case, name_res):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    integral = operations_test_case.ts.interpolate(
        operations_test_case.time_integral, allow_extrapolation=True
    ).integrate(
        integration_constant=operations_test_case.integration_constant_integral,
        **kwargs,
    )

    if name_res is None:
        assert integral.name == f"{operations_test_case.ts.name}_integral"
    else:
        assert integral.name == name_res

    assert isinstance(integral, Timeseries)
    pint.testing.assert_equal(
        operations_test_case.time_integral, integral.time_axis.bounds
    )

    pint.testing.assert_allclose(
        integral.interpolate(
            time_axis=operations_test_case.time_integral
        ).discrete.values_at_bounds.values,
        operations_test_case.exp_integral,
        rtol=1e-10,
    )


@operations_test_cases
@pytest.mark.parametrize("time_axis_arg_raw_pint", (True, False))
def test_interpolate(operations_test_case, time_axis_arg_raw_pint):
    time_interp_raw = operations_test_case.time_interp
    if time_axis_arg_raw_pint:
        time_interp = time_interp_raw
    else:
        time_interp = TimeAxis(time_interp_raw)

    res = operations_test_case.ts.interpolate(time_axis=time_interp)

    assert isinstance(res, Timeseries)

    pint.testing.assert_allclose(
        res.discrete.values_at_bounds.values,
        operations_test_case.exp_interp,
        rtol=1e-10,
    )

    # Check that domain was updated correctly
    for res_v, exp_v in zip(
        (time_interp_raw.min(), time_interp_raw.max()),
        res.timeseries_continuous.domain,
    ):
        pint.testing.assert_equal(res_v, exp_v)

    # Check that times outside time_interp now raise
    with pytest.raises(ExtrapolationNotAllowedError):
        res.interpolate(time_axis=np.atleast_1d(time_interp_raw[0] - Q(1 / 10, "yr")))
    with pytest.raises(ExtrapolationNotAllowedError):
        res.interpolate(time_axis=np.atleast_1d(time_interp_raw[-1] + Q(1 / 10, "yr")))


@operations_test_cases
def test_extrapolate(operations_test_case):
    time_extrap_raw = operations_test_case.time_extrap
    time_axis = TimeAxis(time_extrap_raw)

    res = operations_test_case.ts.interpolate(
        time_axis=time_axis, allow_extrapolation=True
    )

    assert isinstance(res, Timeseries)

    pint.testing.assert_allclose(
        res.discrete.values_at_bounds.values,
        operations_test_case.exp_extrap,
        rtol=1e-10,
    )

    # Check that domain was updated correctly
    for res_v, exp_v in zip(
        (time_extrap_raw.min(), time_extrap_raw.max()),
        res.timeseries_continuous.domain,
    ):
        pint.testing.assert_equal(res_v, exp_v)

    # Check that times outside time_interp now raise
    with pytest.raises(ExtrapolationNotAllowedError):
        res.interpolate(time_axis=np.atleast_1d(time_extrap_raw[0] - Q(1 / 10, "yr")))
    with pytest.raises(ExtrapolationNotAllowedError):
        res.interpolate(time_axis=np.atleast_1d(time_extrap_raw[-1] + Q(1 / 10, "yr")))


def get_test_update_interpolation_self_cases() -> tuple[pytest.param]:
    res = []
    for interp_option, exp_values_at_bounds_same in (
        # All the values effectively get shifted back one time window
        (InterpolationOption.PiecewiseConstantNextLeftClosed, False),
        (InterpolationOption.PiecewiseConstantNextLeftOpen, True),
        (InterpolationOption.PiecewiseConstantPreviousLeftClosed, True),
        # All the values effectively get shifted forward one time window
        (InterpolationOption.PiecewiseConstantPreviousLeftOpen, False),
        (InterpolationOption.Linear, True),
        (InterpolationOption.Quadratic, True),
        (InterpolationOption.Cubic, True),
        (InterpolationOption.Quartic, True),
    ):
        expectation = (
            does_not_raise()
            if exp_values_at_bounds_same
            else pytest.warns(
                InterpolationUpdateChangedValuesAtBoundsWarning,
                match=(
                    f"Updating interpolation to {interp_option.name} "
                    "has caused the values at the bounds defined by "
                    "`self.time_axis` to change."
                ),
            )
        )
        res.append(
            pytest.param(
                interp_option,
                interp_option,
                False,
                exp_values_at_bounds_same,
                {},
                expectation,
                id=f"{interp_option.name}__to__{interp_option.name}",
            )
        )

    return tuple(res)


def get_test_update_interpolation_higher_order_cases() -> tuple[pytest.param]:
    res = []
    for a, b in itertools.combinations(
        (
            InterpolationOption.Linear,
            InterpolationOption.Quadratic,
            InterpolationOption.Cubic,
            InterpolationOption.Quartic,
        ),
        2,
    ):
        for start, end in ((a, b), (b, a)):
            res.append(
                pytest.param(
                    start,
                    end,
                    True,
                    True,
                    {},
                    does_not_raise(),
                    id=f"{start.name}__to__{end.name}",
                )
            )

    return tuple(res)


def get_test_update_interpolation_piecewise_constant_cases() -> tuple[pytest.param]:
    res = []
    for a, b in itertools.combinations(
        (
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftOpen,
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
        ),
        2,
    ):
        for start, end in ((a, b), (b, a)):
            exp_bounds_values_change = (
                end
                in (
                    # The interpolation choices where the interpolated value
                    # at t(i) is y(i + 1), where i is the index in the original
                    # discrete arrays.
                    InterpolationOption.PiecewiseConstantNextLeftClosed,
                    InterpolationOption.PiecewiseConstantPreviousLeftOpen,
                )
                and start != end
            )

            expectation = (
                pytest.warns(
                    InterpolationUpdateChangedValuesAtBoundsWarning,
                    match=(
                        f"Updating interpolation to {end.name} "
                        "has caused the values at the bounds defined by "
                        "`self.time_axis` to change."
                    ),
                )
                if exp_bounds_values_change
                else does_not_raise()
            )

            res.append(
                pytest.param(
                    start,
                    end,
                    True,
                    not exp_bounds_values_change,
                    {},
                    expectation,
                    id=f"{start.name}__to__{end.name}",
                )
            )

    return tuple(res)


def get_test_update_interpolation_piecewise_constant_to_higher_order_cases() -> (
    tuple[pytest.param]
):
    res = []
    for a, b in itertools.product(
        (
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftOpen,
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
        ),
        (
            InterpolationOption.Linear,
            InterpolationOption.Quadratic,
            InterpolationOption.Cubic,
            InterpolationOption.Quartic,
        ),
    ):
        for start, end in ((a, b), (b, a)):
            exp_bounds_values_change = end in (
                # The interpolation choices where the interpolated value
                # at t(i) is y(i + 1), where i is the index in the original
                # discrete arrays.
                InterpolationOption.PiecewiseConstantNextLeftClosed,
                InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            )

            expectation = (
                pytest.warns(
                    InterpolationUpdateChangedValuesAtBoundsWarning,
                    match=(
                        f"Updating interpolation to {end.name} "
                        "has caused the values at the bounds defined by "
                        "`self.time_axis` to change."
                    ),
                )
                if exp_bounds_values_change
                else does_not_raise()
            )

            res.append(
                pytest.param(
                    start,
                    end,
                    True,
                    not exp_bounds_values_change,
                    {},
                    expectation,
                    id=f"{start.name}__to__{end.name}",
                )
            )

    return tuple(res)


@pytest.mark.parametrize(
    [
        "start_interp",
        "end_interp",
        "exp_values_changed",
        "exp_values_at_bounds_same",
        "kwargs",
        "expectation",
    ],
    (
        *get_test_update_interpolation_self_cases(),
        *get_test_update_interpolation_higher_order_cases(),
        *get_test_update_interpolation_piecewise_constant_cases(),
        *get_test_update_interpolation_piecewise_constant_to_higher_order_cases(),
        pytest.param(
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            True,
            False,
            dict(warn_if_values_at_bounds_change=False),
            does_not_raise(),
            id="check_warning_silencing",
        ),
    ),
)
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_update_interpolation(  # noqa: PLR0913
    name_res,
    start_interp,
    end_interp,
    exp_values_changed,
    exp_values_at_bounds_same,
    kwargs,
    expectation,
):
    if name_res is not None:
        kwargs["name_res"] = name_res

    x = Q([1.0, 10.0, 20.0, 30.0, 100.0], "yr")

    ts_start = Timeseries.from_arrays(
        x=x,
        y=Q([10.0, 12.0, 32.0, 20.0, -3.0], "Gt"),
        interpolation=start_interp,
        name="start",
    )

    with expectation:
        res = ts_start.update_interpolation(end_interp, **kwargs)

    assert isinstance(res, Timeseries)
    pint.testing.assert_equal(ts_start.time_axis.bounds, res.time_axis.bounds)

    if name_res is None:
        assert res.name == f"{ts_start.name}_{end_interp.name}"
    else:
        assert res.name == name_res

    if exp_values_at_bounds_same:
        pint.testing.assert_allclose(
            ts_start.discrete.values_at_bounds.values,
            res.discrete.values_at_bounds.values,
            rtol=1e-10,
        )

        if exp_values_changed:
            # We've updated the interpolation so these should change
            check_values_different_times = (
                np.setdiff1d(np.linspace(x.min().m - 10, x.max().m + 10, 100), x.m)
                * x.u
            )
            with pytest.raises(AssertionError):
                start_vals = ts_start.interpolate(
                    check_values_different_times, allow_extrapolation=True
                ).discrete.values_at_bounds.values

                res_vals = res.interpolate(
                    check_values_different_times, allow_extrapolation=True
                ).discrete.values_at_bounds.values

                pint.testing.assert_equal(start_vals, res_vals)

    else:
        # We expect the bounds to have been updated, hence the warning.
        with pytest.raises(AssertionError):
            pint.testing.assert_equal(
                ts_start.discrete.values_at_bounds.values,
                res.discrete.values_at_bounds.values,
            )


def get_test_update_interpolation_integral_preserving_cases() -> tuple[pytest.param]:
    res = []
    for a, b in itertools.combinations(
        (
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftOpen,
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            InterpolationOption.Linear,
            InterpolationOption.Quadratic,
            InterpolationOption.Cubic,
        ),
        2,
    ):
        for start, end in ((a, b), (b, a)):
            # Raise for to piecewise constant except PiecewiseConstantNextLeftClosed
            exp_successful = end not in (
                InterpolationOption.PiecewiseConstantNextLeftOpen,
                InterpolationOption.PiecewiseConstantPreviousLeftClosed,
                InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            )
            res.append(
                pytest.param(
                    start,
                    end,
                    exp_successful,
                    dict(
                        check_change_func=partial(
                            pint.testing.assert_allclose, atol=1e-10
                        )
                    ),
                    id=f"{start.name}__to__{end.name}",
                )
            )

    return tuple(res)


@pytest.mark.parametrize(
    "start_interp, end_interp, exp_successful, kwargs",
    get_test_update_interpolation_integral_preserving_cases(),
)
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_update_interpolation_integral_preserving(
    name_res, start_interp, end_interp, exp_successful, kwargs
):
    if name_res is not None:
        kwargs["name_res"] = name_res

    x = Q([1.0, 10.0, 20.0, 30.0, 100.0], "yr")

    start = Timeseries.from_arrays(
        x=x,
        y=Q([10.0, 12.0, 32.0, 20.0, -3.0], "Gt"),
        interpolation=start_interp,
        name="start",
    )

    expectation = (
        does_not_raise()
        if exp_successful
        else pytest.raises(
            UnreachableIntegralPreservingInterpolationTarget,
            match=(
                f"The interpolation target {end_interp!r} is unreachable "
                "via integral-preserving interpolation. "
                "Please target "
                f"{InterpolationOption.PiecewiseConstantNextLeftClosed!r} "
                "instead."
            ),
        )
    )

    with expectation:
        res = start.update_interpolation_integral_preserving(
            interpolation=end_interp, **kwargs
        )

    if not exp_successful:
        return

    assert isinstance(res, Timeseries)
    pint.testing.assert_equal(start.time_axis.bounds, res.time_axis.bounds)

    if name_res is None:
        assert (
            res.name
            == f"{start.name}_integral-preserving-interpolation-{end_interp.name}"
        )
    else:
        assert res.name == name_res

    # Check integral preserved
    pint.testing.assert_allclose(
        start.integrate(Q(10, "Gt yr")).discrete.values_at_bounds.values,
        res.integrate(Q(10, "Gt yr")).discrete.values_at_bounds.values,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "kwargs, expectation",
    (
        pytest.param(
            dict(
                check_change_func=partial(
                    pint.testing.assert_allclose, atol=0.0, rtol=0.0
                )
            ),
            pytest.warns(InterpolationUpdateChangedValuesAtBoundsWarning),
            id="warning",
        ),
        pytest.param(
            dict(
                check_change_func=partial(
                    pint.testing.assert_allclose, atol=0.0, rtol=0.0
                ),
                warn_if_values_at_bounds_change=False,
            ),
            does_not_raise(),
            id="warning-suppressed",
        ),
    ),
)
def test_update_interpolation_integral_preserving_kwarg_passing(kwargs, expectation):
    x = Q([1.0, 10.0, 20.0, 30.0, 100.0], "yr")

    start = Timeseries.from_arrays(
        x=x,
        y=Q([10.0, 12.0, 32.0, 20.0, -3.0], "Gt"),
        interpolation=InterpolationOption.Linear,
        name="start",
    )

    with expectation:
        start.update_interpolation_integral_preserving(
            interpolation=InterpolationOption.Cubic, **kwargs
        )


@pytest.mark.parametrize(
    "start_interp, exp_values_at_bounds_same",
    (
        (InterpolationOption.PiecewiseConstantNextLeftClosed, True),
        (InterpolationOption.PiecewiseConstantPreviousLeftClosed, True),
        # Open bounds lost upon integration
        (InterpolationOption.PiecewiseConstantNextLeftOpen, False),
        (InterpolationOption.PiecewiseConstantPreviousLeftOpen, False),
        (InterpolationOption.Linear, True),
        (InterpolationOption.Quadratic, True),
        (InterpolationOption.Cubic, True),
        (InterpolationOption.Quartic, True),
    ),
)
def test_integrate_then_differentiate(start_interp, exp_values_at_bounds_same):
    x = Q([1.0, 10.0, 20.0, 30.0, 100.0], "yr")

    start = Timeseries.from_arrays(
        x=x,
        y=Q([10.0, 12.0, 32.0, 20.0, -3.0], "Gt"),
        interpolation=start_interp,
        name="start",
    )

    res = start.integrate(Q(23.0, "Gt yr")).differentiate()

    if exp_values_at_bounds_same:
        pint.testing.assert_allclose(
            res.discrete.values_at_bounds.values,
            start.discrete.values_at_bounds.values,
            rtol=1e-10,
        )

    times_check = np.linspace((2 * x[0] - x[1]).m, (2 * x[-1] - x[-2]).m, 1000) * x.u

    res_check_vals = res.interpolate(
        times_check, allow_extrapolation=True
    ).discrete.values_at_bounds.values
    start_check_vals = start.interpolate(
        times_check, allow_extrapolation=True
    ).discrete.values_at_bounds.values

    pint.testing.assert_allclose(res_check_vals, start_check_vals, rtol=1e-10)


@pytest.mark.parametrize(
    "start_interp, exp_result",
    (
        # Differentiating piecewise constant just leads to zeros
        (InterpolationOption.PiecewiseConstantNextLeftClosed, False),
        (InterpolationOption.PiecewiseConstantPreviousLeftClosed, False),
        (InterpolationOption.PiecewiseConstantNextLeftOpen, False),
        (InterpolationOption.PiecewiseConstantPreviousLeftOpen, False),
        (InterpolationOption.Linear, True),
        (InterpolationOption.Quadratic, True),
        (InterpolationOption.Cubic, True),
        (InterpolationOption.Quartic, True),
    ),
)
def test_differentiate_then_integrate(start_interp, exp_result):
    x = Q([1.0, 10.0, 20.0, 30.0, 100.0], "yr")
    y = Q([10.0, 12.0, 32.0, 20.0, -3.0], "Gt")

    start = Timeseries.from_arrays(
        x=x,
        y=y,
        interpolation=start_interp,
        name="start",
    )

    res = start.differentiate().integrate(y[0])

    times_check = np.linspace((2 * x[0] - x[1]).m, (2 * x[-1] - x[-2]).m, 1000) * x.u

    res_check_vals = res.interpolate(
        times_check, allow_extrapolation=True
    ).discrete.values_at_bounds.values
    start_check_vals = start.interpolate(
        times_check, allow_extrapolation=True
    ).discrete.values_at_bounds.values

    if exp_result:
        pint.testing.assert_allclose(
            res.discrete.values_at_bounds.values,
            start.discrete.values_at_bounds.values,
            rtol=1e-10,
        )

        pint.testing.assert_allclose(res_check_vals, start_check_vals, rtol=1e-10)

    else:
        pint.testing.assert_allclose(res_check_vals, y[0], rtol=1e-10)


@pytest.mark.parametrize(
    "x_units, y_units, plot_kwargs, legend",
    (
        pytest.param(None, None, {}, False, id="no-units-set"),
        pytest.param("month", None, {}, False, id="x-units-set"),
        pytest.param(None, "t", {}, False, id="y-units-set"),
        pytest.param("s", "Gt", {}, False, id="x-and-y-units-set"),
        pytest.param(None, None, {}, True, id="default-labels"),
        pytest.param(
            None,
            None,
            dict(
                continuous_plot_kwargs=dict(label="overwritten-continuous"),
                show_discrete=True,
                discrete_plot_kwargs=dict(label="overwritten-discrete"),
            ),
            True,
            id="overwrite-labels",
        ),
        pytest.param(
            "yr",
            "Gt",
            dict(continuous_plot_kwargs=dict(alpha=0.7, linewidth=2)),
            False,
            id="x-and-y-units-set-kwargs-continuous",
        ),
        pytest.param(
            None,
            None,
            dict(continuous_plot_kwargs=dict(res_increase=2, label="res_increase=2")),
            True,
            id="res-increase",
        ),
        pytest.param(
            None,
            None,
            dict(show_discrete=True),
            True,
            id="show-discrete",
        ),
        pytest.param(
            None,
            None,
            dict(show_discrete=True, show_continuous=False),
            True,
            id="discrete-only",
        ),
        pytest.param(
            None,
            None,
            dict(
                show_discrete=True,
                discrete_plot_kwargs=dict(marker="x", s=150),
                show_continuous=False,
                # Should be ignored
                continuous_plot_kwargs=dict(explode=4),
            ),
            True,
            id="discrete-kwargs",
        ),
        pytest.param(
            None,
            None,
            dict(
                show_discrete=True,
                discrete_plot_kwargs=dict(marker="x", s=150, zorder=3),
                continuous_plot_kwargs=dict(linewidth=2, alpha=0.7),
            ),
            True,
            id="continuous-and-discrete-kwargs",
        ),
    ),
)
def test_plot(  # noqa: PLR0913
    x_units, y_units, plot_kwargs, legend, image_regression, tmp_path
):
    matplotlib = pytest.importorskip("matplotlib")

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.units

    # Setup matplotlib to use units
    UR.setup_matplotlib(enable=True)

    fig, ax = plt.subplots()

    gt = Timeseries.from_arrays(
        x=Q([1.0, 10.0, 20.0], "yr"),
        y=Q([10.0, 12.0, 32.0], "Gt"),
        interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        name="gt_piecewise_constant",
    )

    mt = Timeseries.from_arrays(
        x=Q([0.0, 10.0, 32.0], "yr"),
        y=Q([150.0, 1500.0, 2232.0], "Mt"),
        interpolation=InterpolationOption.Linear,
        name="mt_piecewise_linear",
    )

    gt_per_year = Timeseries.from_arrays(
        x=Q([0.0, 10.0, 32.0], "yr"),
        y=Q([150.0, 1500.0, 2232.0], "Gt / yr"),
        interpolation=InterpolationOption.Linear,
        name="gt_per_year_piecewise_linear",
    )

    if x_units is not None:
        ax.set_xlabel(x_units)
        ax.xaxis.set_units(UR.Unit(x_units))

    if y_units is not None:
        ax.set_ylabel(y_units)
        ax.yaxis.set_units(UR.Unit(y_units))

    # Even though timeseries are in different units,
    # use of pint with matplotib will ensure sensible units on plot.
    mt.plot(ax=ax, **plot_kwargs)
    gt.plot(ax=ax, **plot_kwargs)

    # Trying to plot something with incompatible units will raise.
    with pytest.raises(matplotlib.units.ConversionError):
        gt_per_year.plot(ax=ax, **plot_kwargs)

    if legend:
        ax.legend()

    fig.tight_layout()

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Ensure we tear down
    UR.setup_matplotlib(enable=False)
    plt.close()


@pytest.mark.parametrize(
    "plot_kwargs, expectation",
    (
        pytest.param(
            {},
            pytest.warns(
                UserWarning,
                match=(
                    "The magnitude will be plotted "
                    "without any consideration of units"
                ),
            ),
            id="defaults",
        ),
        pytest.param(
            dict(continuous_plot_kwargs=dict(warn_if_plotting_magnitudes=True)),
            pytest.warns(
                UserWarning,
                match=(
                    "The magnitude will be plotted "
                    "without any consideration of units"
                ),
            ),
            id="warning",
        ),
        pytest.param(
            dict(continuous_plot_kwargs=dict(warn_if_plotting_magnitudes=False)),
            does_not_raise(),
            id="no-warning",
        ),
    ),
)
def test_plot_matplotlib_units_not_registered(
    plot_kwargs, expectation, image_regression, tmp_path
):
    matplotlib = pytest.importorskip("matplotlib")

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ts = Timeseries.from_arrays(
        x=Q([1.0, 10.0, 20.0], "yr"),
        y=Q([10.0, 12.0, 32.0], "Mt / yr"),
        interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        name="piecewise_constant",
    )

    with expectation:
        ts.plot(ax=ax, **plot_kwargs)

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    plt.close()

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="matplotlib_available"),
        pytest.param(
            {"matplotlib": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match="`TimeseriesContinuous.plot` requires matplotlib to be installed",
            ),
            id="matplotlib_not_available",
        ),
    ),
)
def test_plot_ax_creation(sys_modules_patch, expectation):
    pytest.importorskip("matplotlib")

    ts = Timeseries.from_arrays(
        x=Q([1.0, 10.0, 20.0], "yr"),
        y=Q([10.0, 12.0, 32.0], "Mt / yr"),
        interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        name="piecewise_constant",
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            ts.plot(
                continuous_plot_kwargs=dict(
                    warn_if_plotting_magnitudes=False,
                )
            )
