"""
Creation of emissions pathways compatible with a given budget
"""

from __future__ import annotations

import numpy as np

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries import Timeseries
from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
    TimeseriesContinuous,
)
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR


def calculate_linear_net_zero_time(
    budget: PINT_SCALAR,
    budget_start_time: PINT_SCALAR,
    emissions_start: PINT_SCALAR,
) -> PINT_SCALAR:
    """
    Calculate the net-zero time, assuming a linear path to net-zero

    Parameters
    ----------
    budget
        Budget to match

    budget_start_time
        Time from which the budget is available.

        E.g., if the budget is available from 1 Jan 2025,
        supply `Q(2025.0, "yr")` (assuming you're working with pint quantities).

    emissions_start
        Emissions from which the linear path should start

    Returns
    -------
    :
        Net-zero time
    """
    net_zero_time: PINT_SCALAR = budget_start_time + 2 * budget / emissions_start

    return net_zero_time


def derive_linear_path(
    budget: PINT_SCALAR,
    budget_start_time: PINT_SCALAR,
    emissions_start: PINT_SCALAR,
    name_res: str | None = None,
) -> Timeseries:
    """
    Derive a linear pathway that stays within a given budget

    Parameters
    ----------
    budget
        Budget to match

    budget_start_time
        Time from which the budget is available.

        E.g., if the budget is available from 1 Jan 2025,
        supply `Q(2025.0, "yr")` (assuming you're working with pint quantities).

    emissions_start
        Emissions from which the linear path should start

    name_res
        Name to use for the result.

        If not supplied, we use a verbose but clear default.

    Returns
    -------
    :
        Linear pathway to net-zero in line with the budget

    Notes
    -----
    TODO: write out the maths/derivation
    """
    if name_res is None:
        name_res = (
            "Linear emissions\n"
            f"compatible with budget of {budget:.2f}\n"
            f"from {budget_start_time:.2f}"
        )

    net_zero_time = calculate_linear_net_zero_time(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
    )

    last_ts_time = np.floor(net_zero_time) + 2.0 * net_zero_time.to("yr").u

    time_axis_bounds: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [budget_start_time, net_zero_time, last_ts_time]
    )
    values_at_bounds: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [emissions_start, 0.0 * emissions_start, 0.0 * emissions_start]
    )

    emms_linear_pathway = Timeseries.from_arrays(
        time_axis_bounds=time_axis_bounds,
        values_at_bounds=values_at_bounds,
        interpolation=InterpolationOption.Linear,
        name=name_res,
    )

    return emms_linear_pathway


def derive_symmetric_quadratic_path(
    budget: PINT_SCALAR,
    budget_start_time: PINT_SCALAR,
    emissions_start: PINT_SCALAR,
    name_res: str | None = None,
) -> Timeseries:
    """
    Derive a quadratic pathway that stays within a given budget

    The major downside of this approach is
    that the gradient of the output path is zero at `budget_start_time`.

    Parameters
    ----------
    budget
        Budget to match

    budget_start_time
        Time from which the budget is available.

        E.g., if the budget is available from 1 Jan 2025,
        supply `Q(2025.0, "yr")` (assuming you're working with pint quantities).

    emissions_start
        Emissions from which the quadratic path should start.

    name_res
        Name to use for the result.

        If not supplied, we use a verbose but clear default.

    Returns
    -------
    :
        Symmetric quadratic pathway to net-zero in line with the budget

    Notes
    -----
    TODO: write out the maths/derivation
    """
    # Use symmetry argument for derivation
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "derive_symmetric_quadratic_path", requirement="scipy"
        ) from exc

    if name_res is None:
        name_res = (
            "Symmetric quadratic emissions\n"
            f"compatible with budget of {budget:.2f}\n"
            f"from {budget_start_time:.2f}"
        )

    net_zero_time = calculate_linear_net_zero_time(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
    )

    last_ts_time = np.floor(net_zero_time) + 2.0 * net_zero_time.to("yr").u
    time_axis_bounds: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [
            budget_start_time,
            (net_zero_time + budget_start_time) / 2.0,
            net_zero_time,
            last_ts_time,
        ]
    )
    x = time_axis_bounds.to(net_zero_time.u).m

    time_units = budget_start_time.u
    values_units = emissions_start.u
    E_0 = emissions_start
    nzd = (net_zero_time - budget_start_time) / 2.0

    a_coeffs: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [-E_0 / (2.0 * nzd**2), E_0 / (2.0 * nzd**2)]
    )
    b_coeffs: PINT_NUMPY_ARRAY = np.hstack([0.0, -E_0 / nzd])  # type: ignore # mypy confused by pint
    const_terms: PINT_NUMPY_ARRAY = np.hstack([E_0, E_0 / 2.0])  # type: ignore # mypy confused by pint

    c_non_zero: PINT_NUMPY_ARRAY = np.vstack(  # type: ignore # mypy confused by pint
        [
            a_coeffs.to(values_units / time_units**2).m,
            b_coeffs.to(values_units / time_units).m,
            const_terms.to(values_units).m,
        ]
    )
    c = np.hstack([c_non_zero, np.zeros((c_non_zero.shape[0], 1))])

    ppoly = scipy.interpolate.PPoly(c=c, x=x)
    tsc = TimeseriesContinuous(
        name=name_res,
        time_units=time_units,
        values_units=values_units,
        function=ContinuousFunctionScipyPPoly(ppoly),
        domain=(time_axis_bounds.min(), time_axis_bounds.max()),
    )
    emms_quadratic_pathway = Timeseries(
        time_axis=TimeAxis(time_axis_bounds),
        timeseries_continuous=tsc,
    )

    return emms_quadratic_pathway


def convert_to_annual_time_axis(ts: Timeseries) -> Timeseries:
    """
    Convert a timeseries to an annual time axis

    This is just a convenience method.
    It has minimal checks, so may not always produce sensible results.

    Parameters
    ----------
    ts
        Timeseries to convert to an annual time axis.


    Returns
    -------
    :
        `ts`, with its time axis updated so that it has annual steps.

        If `ts`'s first time point is not an integer,
        it is also included in the output time axis
        to ensure that `ts`'s cumulative emissions are conserved.
    """
    annual_time_axis = (
        np.union1d(
            ts.time_axis.bounds.min().to("yr").m,
            np.arange(
                np.ceil(ts.time_axis.bounds.min()).to("yr").m,
                np.ceil(ts.time_axis.bounds.max()).to("yr").m + 1,
                1.0,
            ),
        )
        * ts.time_axis.bounds[0].to("yr").u
    )

    res = ts.interpolate(annual_time_axis, allow_extrapolation=True)

    return res


def convert_to_annual_constant_emissions(
    ts: Timeseries, name_res: str | None = None
) -> Timeseries:
    """
    Convert a timeseries to annual constant emissions

    In other words, to annual-average emissions
    (or annual-total, depending on how you think about emissions),
    like what countries report.

    If the time axis of `ts` starts with an integer year,
    then you can simply sum the output emissions and you will get
    the same cumulative emissions as `ts`.
    If the time axis of `ts` does not start with an integer year,
    then it is more complicated because your first time step
    will not be a full year.

    Parameters
    ----------
    ts
        Timeseries to convert to an annual time axis.

    name_res
        Name to use for the result.

        If not supplied, we use `f"{ts.name}_annualised"`.

    Returns
    -------
    :
        `ts`, with its time axis updated so that it has annual steps
        and is piecewise constant.
    """
    if name_res is None:
        name_res = f"{ts.name}_annualised"

    annual_interp = convert_to_annual_time_axis(ts)
    res = annual_interp.update_interpolation_integral_preserving(
        InterpolationOption.PiecewiseConstantNextLeftClosed, name_res=name_res
    )

    return res
