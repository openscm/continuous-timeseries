"""
Integration tests of `continuous_timeseries.timeseries`
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest
from attrs import define, field, validators
from IPython.lib.pretty import pretty

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries import Timeseries
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR
from continuous_timeseries.values_at_bounds import ValuesAtBounds
from continuous_timeseries.warnings import (
    InterpolationUpdateChangedValuesAtBoundsWarning,
)

UR = pint.get_application_registry()
Q = UR.Quantity


formatting_check_cases = pytest.mark.parametrize(
    "ts",
    (
        pytest.param(
            Timeseries.from_arrays(
                time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
                values_at_bounds=Q([10.0, 12.0, 32.0], "Mt / yr"),
                interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
                name="piecewise_constant",
            ),
            id="piecewise_constant",
        ),
        pytest.param(
            Timeseries.from_arrays(
                time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
                values_at_bounds=Q([10.0, 12.0, 32.0], "Mt / yr"),
                interpolation=InterpolationOption.Linear,
                name="piecewise_linear",
            ),
            id="piecewise_linear",
        ),
        pytest.param(
            Timeseries.from_arrays(
                time_axis_bounds=Q(np.arange(1750.0, 3000.0 + 1), "yr"),
                values_at_bounds=Q(10.0 + np.arange(1251.0), "Mt / yr"),
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
    time_axis_bounds: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )
    values_at_bounds: PINT_NUMPY_ARRAY = field(
        validator=[validators.max_len(3), validators.min_len(3)]
    )

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

    time_interp: PINT_NUMPY_ARRAY
    """Times to use for checking interpolation"""

    exp_interp: PINT_NUMPY_ARRAY
    """Expected values of interpolation at `time_interp`"""

    ts: Timeseries = field()

    @ts.default
    def initialise_timeseries(self):
        return Timeseries.from_arrays(
            time_axis_bounds=self.time_axis_bounds,
            values_at_bounds=self.values_at_bounds,
            interpolation=self.interpolation,
            name=self.name,
        )


#     time_extrap: UR.Quantity
#     """Times to use for checking extrapolation"""
#
#     exp_extrap: UR.Quantity
#     """Expected values of extrapolation at `time_extrap`"""
#
#     time_integral_check: UR.Quantity
#     """Times to use for checking the values of the integral"""
#
#     exp_integral_values_excl_integration_constant: UR.Quantity
#     """
#     Expected values of the derivate at `time_integral_check`
#
#     This excludes the integration constant
#     (i.e. assumes the integration constant is zero).
#     """
#
#     time_derivative_check: UR.Quantity
#     """Times to use for checking the values of the derivative"""
#
#     exp_derivative_values: UR.Quantity
#     """Expected values of the derivate at `time_derivative_check`"""


operations_test_cases = pytest.mark.parametrize(
    "operations_test_case",
    (
        pytest.param(
            OperationsTestCase(
                name="linear",
                interpolation=InterpolationOption.Linear,
                time_axis_bounds=Q([2010, 2020, 2050], "yr"),
                values_at_bounds=Q([-1.0, 0.0, 2.0], "Gt"),
                time_derivative=Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0], "yr"),
                exp_derivative=Q(
                    [
                        1.0 / 10.0,
                        1.0 / 10.0,
                        # On boundary, get the value from the next window
                        # (next closed logic).
                        2.0 / 30.0,
                        2.0 / 30.0,
                        2.0 / 30.0,
                    ],
                    "Gt / yr",
                ),
                time_integral=Q([2010.0, 2020.0, 2030.0, 2050.0], "yr"),
                integration_constant_integral=Q(10.0, "Gt yr"),
                exp_integral=(
                    Q(10.0, "Gt yr")
                    + Q(
                        np.cumsum(
                            [
                                0.0,
                                # y = mx + c
                                # int y dx = mx^2 / 2 + cx + const
                                0.1 * 10.0**2 / 2 - 1 * 10.0,
                                2 / 30.0 * 10.0**2 / 2,
                                2 / 30.0 * 20.0**2 / 2 + 2 / 3 * 20.0,
                            ]
                        ),
                        "Gt yr",
                    )
                ),
                time_interp=Q([2015.0, 2020.0, 2030.0], "yr"),
                exp_interp=Q([-500.0, 0.0, 2000.0 / 3.0], "Mt"),
            ),
            id="linear",
        ),
    ),
)


@operations_test_cases
def test_discrete(operations_test_case):
    exp = TimeseriesDiscrete(
        name=operations_test_case.name,
        time_axis=TimeAxis(operations_test_case.time_axis_bounds),
        # Discrete is interpolated values,
        # not what went in i.e. `operations_test_case.values_at_bounds`
        # (they're not the same thing for piecewise constant in all cases).
        values_at_bounds=ValuesAtBounds(
            operations_test_case.ts.timeseries_continuous.interpolate(
                operations_test_case.time_axis_bounds
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
@pytest.mark.parametrize("name_res", (None, "overwritten"))
def test_differentiate(operations_test_case, name_res):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    derivative = operations_test_case.ts.differentiate(**kwargs)

    if name_res is None:
        assert derivative.name == f"{operations_test_case.ts.name}_derivative"
    else:
        assert derivative.name == name_res

    assert isinstance(derivative, Timeseries)

    pint.testing.assert_equal(
        operations_test_case.ts.time_axis.bounds, derivative.time_axis.bounds
    )

    pint.testing.assert_allclose(
        derivative.interpolate(
            time_axis=operations_test_case.time_derivative
        ).discrete.values_at_bounds.values,
        operations_test_case.exp_derivative,
        rtol=1e-10,
    )


@operations_test_cases
@pytest.mark.parametrize("name_res", (None, "overwritten"))
def test_integrate(operations_test_case, name_res):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    integral = operations_test_case.ts.integrate(
        integration_constant=operations_test_case.integration_constant_integral,
        **kwargs,
    )

    if name_res is None:
        assert integral.name == f"{operations_test_case.ts.name}_integral"
    else:
        assert integral.name == name_res

    assert isinstance(integral, Timeseries)

    pint.testing.assert_equal(
        operations_test_case.ts.time_axis.bounds, integral.time_axis.bounds
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


@pytest.mark.parametrize(
    "start, end, exp_bounds_same, kwargs, expectation",
    (
        pytest.param(
            InterpolationOption.Linear,
            InterpolationOption.Quadratic,
            True,
            {},
            does_not_raise(),
            id="linear_to_quadratic",
        ),
        pytest.param(
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            False,
            {},
            pytest.warns(
                InterpolationUpdateChangedValuesAtBoundsWarning,
                match=(
                    "Updating interpolation to PiecewiseConstantNextLeftClosed "
                    "has caused the values at the bounds defined by "
                    "`self.time_axis` to change."
                ),
            ),
            id="previous_left_closed_to_previous_next_closed",
        ),
        pytest.param(
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            False,
            dict(warn_if_values_at_bounds_change=False),
            does_not_raise(),
            id="previous_left_closed_to_previous_next_closed_warning_silenced",
        ),
    ),
)
def test_update_interpolation_a_to_b(start, end, exp_bounds_same, kwargs, expectation):
    time_axis_bounds = Q([1.0, 10.0, 20.0], "yr")

    start = Timeseries.from_arrays(
        time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
        values_at_bounds=Q([10.0, 12.0, 32.0], "Gt"),
        interpolation=start,
        name="start",
    )

    with expectation:
        res = start.update_interpolation(end, **kwargs)

    if exp_bounds_same:
        pint.testing.assert_equal(
            start.discrete.values_at_bounds.values,
            res.discrete.values_at_bounds.values,
        )

        # We've updated the internal values so these should change
        check_values_different_times = (
            np.setdiff1d(
                np.linspace(time_axis_bounds.min().m, time_axis_bounds.max().m, 100),
                time_axis_bounds.m,
            )
            * time_axis_bounds.u
        )
        with pytest.raises(AssertionError):
            pint.testing.assert_equal(
                start.interpolate(
                    check_values_different_times
                ).discrete.values_at_bounds.values,
                res.interpolate(
                    check_values_different_times
                ).discrete.values_at_bounds.values,
            )

    else:
        # We expect the bounds to have been updated, hence the warning.
        with pytest.raises(AssertionError):
            pint.testing.assert_equal(
                start.discrete.values_at_bounds.values,
                res.discrete.values_at_bounds.values,
            )


# @operations_test_cases
# def test_extrapolate_not_allowed_raises(operations_test_case):
#     with pytest.raises(ExtrapolationNotAllowedError):
#         operations_test_case.ts.interpolate(operations_test_case.time_extrap)
#
#
# @operations_test_cases
# def test_extrapolate(operations_test_case):
#     pint.testing.assert_allclose(
#         operations_test_case.ts.interpolate(
#             operations_test_case.time_extrap, allow_extrapolation=True
#         ),
#         operations_test_case.exp_extrap,
#         rtol=1e-10,
#     )
#
#
# @operations_test_cases
# @pytest.mark.parametrize(
#     "integration_constant",
#     (
#         Q(0, "Gt yr"),
#         Q(1.0, "Gt yr"),
#     ),
# )
# @pytest.mark.parametrize(
#     "integrate_kwargs", ({}, dict(name_res=None), dict(name_res="name_overwritten"))
# )
# def test_integration(operations_test_case, integration_constant, integrate_kwargs):
#     integral = operations_test_case.ts.integrate(
#         integration_constant=integration_constant, **integrate_kwargs
#     )
#
#     pint.testing.assert_allclose(
#         integral.interpolate(operations_test_case.time_integral_check),
#         operations_test_case.exp_integral_values_excl_integration_constant
#         + integration_constant,
#         rtol=1e-10,
#     )
#
#     if (
#         integrate_kwargs
#         and "name_res" in integrate_kwargs
#         and integrate_kwargs["name_res"] is not None
#     ):
#         assert integral.name == integrate_kwargs["name_res"]
#     else:
#         assert integral.name == f"{operations_test_case.ts.name}_integral"


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
    import matplotlib

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.units

    # Setup matplotlib to use units
    UR.setup_matplotlib(enable=True)

    fig, ax = plt.subplots()

    gt = Timeseries.from_arrays(
        time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
        values_at_bounds=Q([10.0, 12.0, 32.0], "Gt"),
        interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        name="gt_piecewise_constant",
    )

    mt = Timeseries.from_arrays(
        time_axis_bounds=Q([0.0, 10.0, 32.0], "yr"),
        values_at_bounds=Q([150.0, 1500.0, 2232.0], "Mt"),
        interpolation=InterpolationOption.Linear,
        name="mt_piecewise_linear",
    )

    gt_per_year = Timeseries.from_arrays(
        time_axis_bounds=Q([0.0, 10.0, 32.0], "yr"),
        values_at_bounds=Q([150.0, 1500.0, 2232.0], "Gt / yr"),
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
    import matplotlib

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ts = Timeseries.from_arrays(
        time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
        values_at_bounds=Q([10.0, 12.0, 32.0], "Mt / yr"),
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
    ts = Timeseries.from_arrays(
        time_axis_bounds=Q([1.0, 10.0, 20.0], "yr"),
        values_at_bounds=Q([10.0, 12.0, 32.0], "Mt / yr"),
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
