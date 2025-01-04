"""
Conversion of discrete to continuous using 'previous' piecewise constant steps

Each interval is open on the left.

In other words,
between t(i) and t(i + 1), the value is equal to y(i).
At t(i + 1), the value is equal to y(i).

If helpful, we have drawn a picture of how this works below.
Symbols:

- time: y-value selected for this time-value
- i: open (i.e. inclusive) boundary
- o: open (i.e. exclusive) boundary

```
y(4):                                                oxxxxxxxxxxxxxx
y(3):                                    oxxxxxxxxxxxi
y(2):                        oxxxxxxxxxxxi
y(1): xxxxxxxxxxxxxxxxxxxxxxxi
      -----------|-----------|-----------|-----------|--------------
              time(1)     time(2)     time(3)     time(4)
```
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from attrs import define, field

from continuous_timeseries.discrete_to_continuous.piecewise_constant_common import (
    differentiate_piecewise_constant,
    discrete_to_continuous_piecewise_constant,
    get_values,
    integrate_piecewise_constant,
    piecewise_constant_y_validator,
)
from continuous_timeseries.typing import NP_ARRAY_OF_FLOAT_OR_INT, NP_FLOAT_OR_INT

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import ContinuousFunctionScipyPPoly


def get_idxs(
    times: NP_ARRAY_OF_FLOAT_OR_INT, self_x: NP_ARRAY_OF_FLOAT_OR_INT
) -> npt.NDArray[np.integer]:
    """
    Get the indexes from `self.y` to return, given the times of interest and `self.x`

    This function defines the key logic of the interpolation implementation.

    Parameters
    ----------
    times
        Times for which to retrieve the values

    self_x
        The points which define the piecewise constant intervals

    Returns
    -------
    :
        The indexes from `self.y` to return
    """
    res_idxs: npt.NDArray[np.integer] = (
        np.searchsorted(a=self_x, v=np.atleast_1d(times), side="left") - 1
    )
    # Fix up any underrun
    res_idxs[res_idxs == -1] = 0

    return res_idxs


@define
class PPolyPiecewiseConstantPreviousLeftOpen:
    """
    Piecewise polynomial that implements our 'previous' constant left-open logic

    For full details of the logic, see [the module's docstring][(m)].

    We can't use [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly] directly
    because it doesn't behave as we want at the first boundary.
    We could subclass [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly],
    but that is more trouble than its worth for such a simple implementation.
    """

    x: NP_ARRAY_OF_FLOAT_OR_INT
    """
    Breakpoints between each piecewise constant interval
    """

    y: NP_ARRAY_OF_FLOAT_OR_INT = field(validator=[piecewise_constant_y_validator])
    """
    The y-values which help define our spline.

    Note that these are not the same as the values at our boundaries,
    see [the module's docstring][(m)].
    """

    def __call__(
        self, x: NP_ARRAY_OF_FLOAT_OR_INT, allow_extrapolation: bool = False
    ) -> NP_ARRAY_OF_FLOAT_OR_INT:
        """
        Evaluate the function at specific points

        Parameters
        ----------
        x
            Points at which to evaluate the function

        allow_extrapolation
            Should extrapolation be allowed?

        Returns
        -------
        :
            The function, evaluated at `x`

        Raises
        ------
        ExtrapolationNotAllowedError
            The user attempted to extrapolate when it isn't allowed.
        """
        res = get_values(
            times=x,
            self_x=self.x,
            self_y=self.y,
            get_idxs=get_idxs,
            allow_extrapolation=allow_extrapolation,
        )

        return res

    def differentiate(self) -> ContinuousFunctionScipyPPoly:
        """
        Differentiate

        Returns
        -------
        :
            Derivative of the function
        """
        res = differentiate_piecewise_constant(self=self)

        return res

    def integrate(
        self, integration_constant: NP_FLOAT_OR_INT, domain_start: NP_FLOAT_OR_INT
    ) -> ContinuousFunctionScipyPPoly:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant

            This is required for the integral to be a definite integral.

        domain_start
            The start of the domain.

            This is required to ensure that we start at the right point
            when evaluating the definite integral.

        Returns
        -------
        :
            Integral of the function
        """
        res = integrate_piecewise_constant(
            self=self,
            integration_constant=integration_constant,
            domain_start=domain_start,
        )

        return res


discrete_to_continuous_piecewise_constant_previous_left_open = partial(
    discrete_to_continuous_piecewise_constant,
    piecewise_constant_like=PPolyPiecewiseConstantPreviousLeftOpen,
)
