"""
Integration tests of `continuous_timeseries.budget_compatible_pathways`
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint.testing
import pytest

from continuous_timeseries import InterpolationOption, Timeseries
from continuous_timeseries import budget_compatible_pathways as ct_bcp
from continuous_timeseries.exceptions import MissingOptionalDependencyError

openscm_units = pytest.importorskip("openscm_units")

UR = openscm_units.unit_registry
Q = UR.Quantity


budget_cases = pytest.mark.parametrize(
    "budget, budget_start_time, emissions_start",
    (
        pytest.param(
            Q(500, "GtC"),
            Q(2020, "yr"),
            Q(10.2, "GtC / yr"),
            id="ar6_global_budget_like",
        ),
        pytest.param(
            Q(1.5, "GtCO2"),
            Q(2025, "yr"),
            Q(453.2, "MtCO2 / yr"),
            id="country_small_budget_like",
        ),
        pytest.param(
            Q(15.0, "GtCO2"),
            Q(2025, "yr"),
            Q(745.0, "MtCO2 / yr"),
            id="country_bigger_budget_like",
        ),
    ),
)


@budget_cases
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_budget_compatible_linear_pathway_derivation(
    name_res,
    budget,
    budget_start_time,
    emissions_start,
):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    res = ct_bcp.derive_linear_path(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
        **kwargs,
    )

    if name_res is None:
        assert res.name == (
            "Linear emissions\n"
            f"compatible with budget of {budget:.2f}\n"
            f"from {budget_start_time:.2f}"
        )

    else:
        assert res.name == name_res

    # First value should be input emissions
    discrete_values = res.discrete.values_at_bounds.values
    pint.testing.assert_equal(discrete_values[0], emissions_start)

    # Last two values should be zero
    pint.testing.assert_equal(discrete_values[-2:], 0.0 * emissions_start)

    # Derivative should be constant (i.e. function is linear),
    # except for the last two timesteps,
    # which should both have a gradient of zero.
    derivative_values = (
        ct_bcp.convert_to_annual_time_axis(res)
        .differentiate()
        .discrete.values_at_bounds.values
    )
    pint.testing.assert_equal(derivative_values[:-2], derivative_values[0])
    pint.testing.assert_equal(derivative_values[-2:], derivative_values[0] * 0.0)

    # Ensure that we budget is respected
    pint.testing.assert_allclose(
        res.integrate(
            Q(0, emissions_start.u * budget_start_time.u)
        ).discrete.values_at_bounds.values[-1],
        budget,
    )
    # Ensure that extrapolating post the net-zero results in zeros
    net_zero_year = res.time_axis.bounds[1]
    res_extrap = res.interpolate(
        np.hstack(
            [net_zero_year, net_zero_year + 3 * (net_zero_year - budget_start_time)]
        ),
        allow_extrapolation=True,
    )
    pint.testing.assert_equal(
        res_extrap.discrete.values_at_bounds.values, 0.0 * emissions_start
    )


@budget_cases
@pytest.mark.parametrize(
    "name_res",
    (
        pytest.param(None, id="default_name_res"),
        pytest.param("overwritten", id="name_res_supplied"),
    ),
)
def test_budget_compatible_symmetric_quadratic_pathway_derivation(
    name_res,
    budget,
    budget_start_time,
    emissions_start,
):
    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    res = ct_bcp.derive_symmetric_quadratic_path(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
        **kwargs,
    )

    if name_res is None:
        assert res.name == (
            "Symmetric quadratic emissions\n"
            f"compatible with budget of {budget:.2f}\n"
            f"from {budget_start_time:.2f}"
        )

    else:
        assert res.name == name_res

    # First value should be input emissions
    discrete_values = res.discrete.values_at_bounds.values
    pint.testing.assert_equal(discrete_values[0], emissions_start)

    # Last two values should be zero
    pint.testing.assert_equal(discrete_values[-2:], 0.0 * emissions_start)

    # Derivative should start at zero,
    # then climb to its peak,
    # then be back at zero by the net-zero year (two quadratics).
    net_zero_year = res.time_axis.bounds[-2]
    derivative_times = np.hstack(
        [
            budget_start_time,
            (budget_start_time + net_zero_year) / 2.0,
            net_zero_year,
        ]
    )
    derivative_values = (
        res.differentiate()
        .interpolate(derivative_times)
        .discrete.values_at_bounds.values
    )
    # Zero at start and end
    pint.testing.assert_equal(0.0 * derivative_values[0], derivative_values[[0, -1]])
    # Peak gradient
    nzd = (net_zero_year - budget_start_time) / 2.0
    exp_peak_gradient = -emissions_start / nzd
    pint.testing.assert_allclose(derivative_values.min(), exp_peak_gradient)

    # Two symmetric quadratics.
    exp_a = emissions_start / (2.0 * nzd**2)
    exp_second_derivative_abs = 2.0 * exp_a
    second_derivative_times = (
        np.union1d(
            derivative_times.to("yr").m,
            (
                np.linspace(
                    budget_start_time.to("yr").m,
                    net_zero_year.to("yr").m,
                    100,
                )
            ),
        )
        * budget_start_time.to("yr").u
    )
    second_derivative_values = (
        res.differentiate()
        .differentiate()
        .interpolate(second_derivative_times)
        .discrete.values_at_bounds.values
    )
    pre_mid_point = second_derivative_times < (budget_start_time + net_zero_year) / 2.0
    pint.testing.assert_allclose(
        second_derivative_values[pre_mid_point], -exp_second_derivative_abs
    )
    pint.testing.assert_allclose(
        second_derivative_values[~pre_mid_point][:-1], exp_second_derivative_abs
    )
    # At net-zero year, second derivative is zero
    pint.testing.assert_allclose(
        second_derivative_values[-1], 0.0 * exp_second_derivative_abs
    )

    # Ensure that we budget is respected
    pint.testing.assert_allclose(
        res.integrate(
            Q(0, emissions_start.u * budget_start_time.u)
        ).discrete.values_at_bounds.values[-1],
        budget,
    )
    # Ensure that extrapolating post the net-zero results in zeros
    res_extrap = res.interpolate(
        np.hstack(
            [net_zero_year, net_zero_year + 3 * (net_zero_year - budget_start_time)]
        ),
        allow_extrapolation=True,
    )
    pint.testing.assert_equal(
        res_extrap.discrete.values_at_bounds.values, 0.0 * emissions_start
    )


# ensure convert_to_annual_constant_emissions works
# for integer and non-integer start year
# (integer start, just get steps)
# ()
@pytest.mark.parametrize(
    "start_time_axis, exp_out_time_axis",
    (
        pytest.param(
            Q([2010.0, 2029.5, 2030.0], "yr"),
            Q(np.arange(2010, 2030 + 1), "yr"),
            id="integer_start_year_integer_end_year",
        ),
        pytest.param(
            Q([2010.5, 2050.3, 2052.0], "yr"),
            Q(
                np.hstack([2010.5, np.arange(2011.0, 2052 + 0.1)]),
                "yr",
            ),
            id="non_integer_start_year_integer_end_year",
        ),
        pytest.param(
            Q([2010.0, 2050.3, 2052.5], "yr"),
            Q(np.arange(2010, 2053 + 1), "yr"),
            id="integer_start_year_non_integer_end_year",
        ),
        pytest.param(
            Q([2010.5, 2050.0, 2050.5], "yr"),
            Q(
                np.hstack([2010.5, np.arange(2011.0, 2051 + 0.1)]),
                "yr",
            ),
            id="non_integer_start_year_non_integer_end_year",
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
def test_convert_to_annual_constant_emissions(
    name_res, start_time_axis, exp_out_time_axis
):
    name_start = "name_start"

    kwargs = {}
    if name_res is not None:
        kwargs["name_res"] = name_res

    start = Timeseries.from_arrays(
        x=start_time_axis,
        y=Q([1.0, 0.0, 0.0], "MtCO2 / yr"),
        interpolation=InterpolationOption.Linear,
        name=name_start,
    )

    res = ct_bcp.convert_to_annual_constant_emissions(start, **kwargs)

    if name_res is not None:
        assert res.name == name_res
    else:
        assert res.name == f"{start.name}_annualised"

    pint.testing.assert_equal(res.time_axis.bounds, exp_out_time_axis)

    # Result should be piecewise constant, hence
    pint.testing.assert_allclose(
        res.differentiate().discrete.values_at_bounds.values, Q(0.0, "MtCO2 / yr / yr")
    )

    # Check integral preserved
    exp_integral = start.integrate(Q(0.0, "MtCO2")).discrete.values_at_bounds.values[-1]
    pint.testing.assert_allclose(
        exp_integral,
        res.integrate(Q(0.0, "MtCO2")).discrete.values_at_bounds.values[-1],
    )
    if res.time_axis.bounds.m[0] == int(res.time_axis.bounds.m[0]):
        # Make sure that simply adding up values gives back our budget
        np.testing.assert_allclose(
            exp_integral,
            np.cumsum(res.discrete.values_at_bounds.values)
            .to(exp_integral.u / res.time_axis.bounds.u)
            .m[-1],
            rtol=1e-3,
        )

    # else:
    #     # In this case, the awkward half year means that simple addition doesn't work.
    #     # There is nothing we can do about that.


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="scipy_available"),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=(
                    "`derive_symmetric_quadratic_path` "
                    "requires scipy to be installed"
                ),
            ),
            id="scipy_not_available",
        ),
    ),
)
def test_no_scipy_quadratic(sys_modules_patch, expectation):
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            ct_bcp.derive_symmetric_quadratic_path(
                budget=Q(450.0, "GtC"),
                budget_start_time=Q(2010.0, "yr"),
                emissions_start=Q(2.3, "GtC / yr"),
            )
