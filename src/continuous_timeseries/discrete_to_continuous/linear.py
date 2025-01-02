"""
Conversion of discrete to continuous data assuming linear interpolation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from continuous_timeseries.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


def discrete_to_continuous_linear(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "ContinuousFunctionScipyPPoly.integrate", requirement="scipy"
        ) from exc

    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )

    time_bounds = discrete.time_axis.bounds
    x = time_bounds.m
    time_steps = time_bounds[1:] - time_bounds[:-1]

    coeffs = np.zeros((2, discrete.values_at_bounds.values.size - 1))

    all_vals = discrete.values_at_bounds.values

    rises = all_vals[1:] - all_vals[:-1]

    coeffs[0, :] = (rises / time_steps).m
    coeffs[1, :] = all_vals.m[:-1]

    piecewise_polynomial = scipy.interpolate.PPoly(
        x=x,
        c=coeffs,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=time_bounds.u,
        values_units=all_vals.u,
        function=ContinuousFunctionScipyPPoly(piecewise_polynomial),
    )

    return res
