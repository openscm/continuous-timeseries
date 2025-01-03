"""
Conversion of discrete to continuous data using 'next' piecewise constant steps

Each interval is closed on the left.

In other words,
between t(i) and t(i + 1), the value is equal to y(i + 1).
At t(i), the value is equal to y(i + 1).

If helpful, we have drawn a picture of how this works below.
Symbols:

- time: y-value selected for this time-value
- i: closed (i.e. inclusive) boundary
- o: open (i.e. exclusive) boundary

```
y(4):                                    ixxxxxxxxxxxxxxxxxxxxxxxxxx
y(3):                        ixxxxxxxxxxxo
y(2):            ixxxxxxxxxxxo
y(1): xxxxxxxxxxxo
      -----------|-----------|-----------|-----------|--------------
              time(1)     time(2)     time(3)     time(4)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from continuous_timeseries.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


def discrete_to_continuous_piecewise_constant_next_left_closed(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise constant

    Here we use piecewise constant, next, left-closed interpolation.
    For details, see
    [the module's docstring][continuous_timeseries.discrete_to_continuous.piecewise_constant_next_left_closed].

    Parameters
    ----------
    discrete
        Discrete timeseries to convert

    Returns
    -------
    :
        Continuous version of `discrete`
        based on piecewise constant, next, left-closed interpolation.
    """  # noqa: E501
    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "discrete_to_continuous_piecewise_constant_next_left_closed",
            requirement="scipy",
        ) from exc

    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )

    time_bounds = discrete.time_axis.bounds
    x = time_bounds.m

    all_vals = discrete.values_at_bounds.values
    # Next left closed, so we can ignore the first value
    coeffs = np.atleast_2d(all_vals[1:].m)

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
