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
from continuous_timeseries.typing import PINT_NUMPY_ARRAY

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous


def discrete_to_continuous_higher_order(
    x: PINT_NUMPY_ARRAY, y: PINT_NUMPY_ARRAY, name: str, order: int
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise higher-order polynomial.

    Here, higher-order means quadratic or higher.
    For details, see
    [the module's docstring][continuous_timeseries.discrete_to_continuous.higher_order].

    Parameters
    ----------
    x
        The discrete x-values from which to convert

    y
        The discrete y-values from which to convert

    name
        The value to use to set the result's name attribute

    order
        Order of the polynomial to fit and return.

    Returns
    -------
    :
        Continuous version of `discrete`
        based on piecewise interpolation of order `order`.
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

    # This is the bit that can go very wrong if done blindly.
    # Hence why this function is only a convenience.
    tck = scipy.interpolate.splrep(x=x.m, y=y.m, k=order)
    piecewise_polynomial = scipy.interpolate.PPoly.from_spline(tck)

    res = TimeseriesContinuous(
        name=name,
        time_units=x.u,
        values_units=y.u,
        function=ContinuousFunctionScipyPPoly(piecewise_polynomial),
        domain=(np.min(x), np.max(x)),
    )

    return res
