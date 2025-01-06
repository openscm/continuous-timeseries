"""
Conversion of discrete to continuous data assuming linear interpolation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.typing import PINT_NUMPY_ARRAY

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous


def discrete_to_continuous_linear(
    x: PINT_NUMPY_ARRAY,
    y: PINT_NUMPY_ARRAY,
    name: str,
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to a piecewise linear

    For details, see
    [the module's docstring][continuous_timeseries.discrete_to_continuous.linear].

    Parameters
    ----------
    x
        The discrete x-values from which to convert

    y
        The discrete y-values from which to convert

    name
        The value to use to set the result's name attribute

    Returns
    -------
    :
        Continuous version of `discrete`
        based on a piecewise linear interpolation.
    """
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "discrete_to_continuous_linear", requirement="scipy"
        ) from exc

    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )

    x_m = x.m
    time_steps = x[1:] - x[:-1]

    coeffs = np.zeros((2, y.size - 1))

    rises = y[1:] - y[:-1]

    coeffs[0, :] = (rises / time_steps).m
    coeffs[1, :] = y.m[:-1]

    piecewise_polynomial = scipy.interpolate.PPoly(
        x=x_m,
        c=coeffs,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=name,
        time_units=x.u,
        values_units=y.u,
        function=ContinuousFunctionScipyPPoly(piecewise_polynomial),
        domain=(np.min(x), np.max(x)),
    )

    return res
