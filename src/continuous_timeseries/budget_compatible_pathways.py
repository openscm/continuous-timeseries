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
    r"""
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
    We're solving for emissions, $y$, as a function of time, $x$.
    We pick a linear emissions pathway

    $$
    y(x) = e_0 * (1 - \frac{x - x_0}{x_{nz} - x_0})
    $$

    Simplifying slightly, we have

    $$
    y(x) = e_0 * \frac{x_{nz} - x}{x_{nz} - x_0}
    $$

    where $e_0$ is emissions at the known time (normally today), $x_0$,
    and $x_{nz}$ is the net-zero time.

    By geometry, the integral of this curve between $x_0$ and $x_nz$ is:

    $$
    \frac{1}{2} (x_0 - x_{nz}) * e_0
    $$

    You can also do this with calculus:

    $$
    \begin{equation}
    \begin{split}
    \int_{x_0}^{x_{nz}} y(x) \, dx &= \int_{x_0}^{x_{nz}} e_0 * (x_{nz} - x) / (x_{nz} - x_0) \, dx \\
                             &= [-e_0 (x_{nz} - x)^2 / (2 * (x_{nz} - x_0))]_{x_0}^{x_{nz}} \\
                             &= -e_0 (x_{nz} - x_0)^2 / (2 * (x_{nz} - x_0))) - e_0 (x_{nz} - x_{nz})^2 / (2 * (x_{nz} - x_0))) \\
                             &= e_0 (x_0 - x_{nz}) / 2
    \end{split}
    \end{equation}
    $$

    This integral should be equal to the allowed buget:

    $$
    \frac{1}{2} (x_0 - x_{nz}) * e_0 = \text{budget}
    $$

    therefore

    $$
    x_{nz} = x_0 + \frac{2 * \text{budget}}{e_0}
    $$
    """  # noqa: E501
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

    x_res: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [budget_start_time, net_zero_time, last_ts_time]
    )
    y_res: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [emissions_start, 0.0 * emissions_start, 0.0 * emissions_start]
    )

    emms_linear_pathway = Timeseries.from_arrays(
        x=x_res,
        y=y_res,
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
    r"""
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
    We're solving for emissions, $y$, as a function of time, $x$.
    We pick a quadratic emissions pathway comprised of two pieces.
    The two pieces are equal in size and split the time period
    from the known time (normally today), $x_0$,
    to the net zero time, $x_{nz}$, in two.
    For convenience, we also define the halfway to net-zero point,
    $x_{nzh} = \frac{x_0 + x_{nz}}{2}$.

    $$
    y(x) = \begin{cases}
       a_1 (x - x_0)^2 + b_1 (x - x_0) + c_1 &\text{if } x \leq x_{nzh} \\
       a_2 (x - x_{nzh})^2 + b_2 (x - x_{nzh}) + c_2 &\text{otherwise}
    \end{cases}
    $$

    Therefore

    $$
    \frac{dy}{dx}(x) = \begin{cases}
       2 a_1 (x - x_0) + b_1 &\text{if } x \leq x_{nzh} \\
       2 a_2 (x - x_{nzh}) + b_2 &\text{otherwise}
    \end{cases}
    $$

    To set the constants, we need some boundary conditions.

    At $x = x_0$, emissions should be equal to the starting emissions, $e_0$.
    This immediately sets $c_1 = e_0$.

    We also choose to set the quadratics such that
    at the time point which is halfway to net zero,
    emissions are half of their original value.
    Thus, at $x = x_{nzh}$, emissions should be $e_0 / 2$.
    This immediately sets $c_2 = e_0 / 2$.

    This condition also means that the decrease in emissions should be the same
    between $x_0$ and $x_{nzh}$ as it is between $x_{nzh}$ and $x_{nz}$.
    We also want the gradient to be continuous
    at the boundary between the two quadratics.

    As a result, we can see that the two quadratics are simply
    a translation and a reflection of each other
    (they have the same change between their defining points
    and the same gradient at the boundary between the two intervals,
    so there are no more degrees of freedom with which we could
    introduce different shapes),
    i.e. they are symmetric (about a carefully chosen axis).

    By the symmetry argument, we have that $a_1 = -a_2$
    and the gradient at $x=x_0$ must equal the gradient at $x=x_{nx}$.
    The gradient at $x=x_{nx}$ is zero, therefore

    $$
    \frac{dy}{dx}(x_0) = 2 a_1 (x_0 - x_0) + b_1 = 0.0 \Rightarrow b_1 = 0.0
    $$

    We can now use the fact that $y(x_{nzh}) = e_0 / 2$ to solve for $a_1$,
    and therefore also $a_2$.

    $$
    \begin{equation}
    \begin{split}
    y(x_{nzh}) &= e_0 / 2 \\
    a_1 (x_{nzh} - x_0)^2 + e_0 &= e_0 / 2 \\
    a_1 &= -\frac{e_0}{2 (x_{nzh} - x_0)^2}
    \end{split}
    \end{equation}
    $$

    The last constant to solve for is $b_2$.
    We solve for this using the first-order continuity constraint at the boundary.

    $$
    \frac{dy}{dx}(x_{nzh}) = 2 a_1 (x_{nzh} - x_0) = 2 a_2 (x - x_{nzh}) + b_2 \Rightarrow b_2 = 2 a_1 (x_{nzh} - x_0)
    $$

    In summary, our constants are

    $$
    \begin{equation}
    \begin{split}
    a_1 &= -\frac{e_0}{2 (x_{nzh} - x_0)^2} \\
    a_2 &= -a_1 \\
    b_1 &= 0.0 \\
    b_2 &= 2 a_1 (x_{nzh} - x_0) \\
    c_1 &= e_0 \\
    c_2 &= e_0 / 2
    \end{split}
    \end{equation}
    $$

    The last question is, where does the net-zero year come from?
    To answer this, first consider the integral of our function
    from $x_0$ to $x_{nz}$

    $$
    \begin{equation}
    \begin{split}
    \int_{x_0}^{x_{nz}} y(x) \, dx &= \int_{x_0}^{x_{nzh}} a_1 (x - x_0)^2 + c_1 \, dx \\
                                    & \quad + \int_{x_{nzh}}^{x_{nz}} a_2 (x - x_{nzh})^2 + b_2 (x - x_{nzh}) + c_2 \, dx \\
                                   &= [ a_1 (x - x_0)^3 / 3 + c_1 x ]_{x_0}^{x_{nzh}} \\
                                    & + [ a_2 (x - x_{nzh})^3 / 3 + b_2 (x - x_{nzh})^2 / 2 + c_2 x ]_{x_{nzh}}^{x_{nz}} \\
                                   &= a_1 (x_{nzh} - x_0)^3 / 3 + c_1 (x_{nzh} - x_0) \\
                                    & + a_2 (x_{nz} - x_{nzh})^3 / 3 + b_2 (x_{nz} - x_{nzh})^2 / 2 + c_2 (x_{nz} - x_{nzh})
    \end{split}
    \end{equation}
    $$

    We next note that

    $$
    x_{nzh} - x_0 = \frac{x_0 + x_{nz}}{2} - x_0 = \frac{x_{nz} - x_0}{2} = \frac{2 x_{nz} - (x_{nz} + x_0)}{2} = x_{nz} - x_{nzh}
    $$

    Hence the cubic terms cancel because $a_1 = -a_2$ and we are left with

    $$
    \begin{equation}
    \begin{split}
    \int_{x_0}^{x_{nz}} y(x) \, dx &= c_1 (x_{nzh} - x_0) + b_2 (x_{nz} - x_{nzh})^2 / 2 + c_2 (x_{nzh} - x_0) \\
                                   &= e_0 (x_{nzh} - x_0) + 2 a_1 (x_{nzh} - x_0) (x_{nz} - x_{nzh})^2 / 2 + e_0 / 2 (x_{nzh} - x_0) \\
                                   &= \frac{3}{2} e_0 (x_{nzh} - x_0) - 2 \frac{e_0}{2 (x_{nzh} - x_0)^2} (x_{nzh} - x_0) (x_{nz} - x_{nzh})^2 / 2 \\
                                   &= \frac{3}{2} e_0 (x_{nzh} - x_0) - \frac{e_0}{2 (x_{nzh} - x_0)} (x_{nz} - x_{nzh})^2 \\
                                   &= \frac{3}{2} e_0 (x_{nzh} - x_0) - \frac{e_0}{2 (x_{nzh} - x_0)} (x_{nzh} - x_0)^2 \\
                                   &= \frac{3}{2} e_0 (x_{nzh} - x_0) - \frac{e_0}{2} (x_{nzh} - x_0) \\
                                   &= \frac{3}{2} e_0 (x_{nzh} - x_0) - \frac{e_0}{2} (x_{nzh} - x_0) \\
                                   &= e_0 (x_{nzh} - x_0) \\
                                   &= e_0 (\frac{x_{nz} - x_0}{2}) \\
                                   &= \frac{1}{2} e_0 (x_{nz} - x_0)
    \end{split}
    \end{equation}
    $$

    Put more simply,
    the integral is simply equal to the integral of a straight-line to net-zero.
    Hence, we can simply use the net-zero year of a linear pathway to net-zero
    in line with the budget,
    and our quadratic pathway will have the same cumulative emissions
    (i.e. will also match the budget).

    As a result, our recipe is:

    1. Calculate the net-zero year of a straight-line pathway to net-zero year.
    1. Use that net-zero year to calculate our constants.
    """  # noqa: E501
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

    time_units = budget_start_time.u
    values_units = emissions_start.u

    net_zero_time = calculate_linear_net_zero_time(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
    )

    e_0 = emissions_start
    x_0 = budget_start_time
    x_nz = net_zero_time
    x_nzh = (x_0 + x_nz) / 2.0

    a_1 = -e_0 / (2.0 * (x_nzh - x_0) ** 2)
    a_2 = -a_1
    b_1 = 0.0 * emissions_start.u / budget_start_time.u
    b_2 = 2.0 * a_1 * (x_nzh - x_0)
    c_1 = e_0
    c_2 = e_0 / 2.0

    a_coeffs: PINT_NUMPY_ARRAY = np.hstack([a_1, a_2])  # type: ignore # mypy confused ay pint
    b_coeffs: PINT_NUMPY_ARRAY = np.hstack([b_1, b_2])  # type: ignore # mypy confused by pint
    const_terms: PINT_NUMPY_ARRAY = np.hstack([c_1, c_2])  # type: ignore # mypy confused by pint

    c_non_zero: PINT_NUMPY_ARRAY = np.vstack(  # type: ignore # mypy confused by pint
        [
            a_coeffs.to(values_units / time_units**2).m,
            b_coeffs.to(values_units / time_units).m,
            const_terms.to(values_units).m,
        ]
    )

    # Ensure we have a zero tail
    last_ts_time = np.floor(net_zero_time) + 2.0 * net_zero_time.to("yr").u
    time_axis_bounds: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by pint
        [x_0, x_nzh, x_nz, last_ts_time]
    )

    x = time_axis_bounds.to(net_zero_time.u).m
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
