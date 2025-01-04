"""
Conversion of discrete to continuous data assuming higher-order interpolation

Here, "higher-order" means quadratic or higher.

In general, this sort of interpolation is tricky and can easily go wrong.
This module is intended as a convenience.
However, in most cases, you will want to use the lower-level interfaces
more directly so you have better control of the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from continuous_timeseries.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


def discrete_to_continuous_higher_order(
    discrete: TimeseriesDiscrete, order: int
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise higher-order polynomial.

    Here, higher-order means quadratic or higher.
    For details, see
    [the module's docstring][continuous_timeseries.discrete_to_continuous.higher_order].

    Parameters
    ----------
    discrete
        Discrete timeseries to convert

    order
        Order of the polynomial to return.

    Returns
    -------
    :
        Continuous version of `discrete`
        based on piecewise a `order` interpolation.
    """
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "discrete_to_continuous_higher_order", requirement="scipy"
        ) from exc

    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )

    time_bounds = discrete.time_axis.bounds
    all_vals = discrete.values_at_bounds.values

    x = time_bounds.m
    y = all_vals.m

    # This is the bit that can go very wrong if done blindly.
    # Hence why this function is only a convenience.
    tck = scipy.interpolate.splrep(x=x, y=y, k=order)
    piecewise_polynomial = scipy.interpolate.PPoly.from_spline(tck)

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=time_bounds.u,
        values_units=all_vals.u,
        function=ContinuousFunctionScipyPPoly(piecewise_polynomial),
        domain=(np.min(time_bounds), np.max(time_bounds)),
    )

    return res
