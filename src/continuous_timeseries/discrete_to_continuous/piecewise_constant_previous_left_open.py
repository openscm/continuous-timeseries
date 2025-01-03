"""
Conversion of discrete to continuous using 'previous' piecewise constant steps

Each interval is open on the left.

In other words,
between t(i) and t(i + 1), the value is equal to y(i).
At t(i + 1), the value is equal to y(i).

If helpful, we have drawn a picture of how this works below.
Symbols:

- time: y-value selected for this time-value
- i: closed (i.e. inclusive) boundary
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

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from attrs import define

from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.typing import NP_FLOAT_OR_INT

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


@define
class PPolyPiecewiseConstantPreviousLeftOpen:
    """
    Piecewise polynomial that implements our 'previous' constant left-open logic

    We can't use [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly] directly
    because it doesn't behave as we want (it does 'next' logic).
    We could subclass [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly],
    but that is more trouble than its worth for such a simple implementation.
    """

    x: npt.NDArray[NP_FLOAT_OR_INT]
    """
    Breakpoints between each piecewise constant interval
    """

    values: npt.NDArray[NP_FLOAT_OR_INT]
    """
    Value to return in each interval.

    Must have same number of elements as `x`
    so that we know what to use for extrapolation beyond the last boundary too.
    """

    # TODO: add validation

    def __call__(
        self, x: npt.NDArray[NP_FLOAT_OR_INT], allow_extrapolation: bool = False
    ) -> npt.NDArray[NP_FLOAT_OR_INT]:
        """
        Evaluate the function at specific points

        Parameters
        ----------
        x
            Points at which to evaluate the function

        allow_extrapolation
            Should extrapolatino be allowed?

        Returns
        -------
        :
            The function, evaluated at `x`

        Raises
        ------
        ExtrapolationNotAllowedError
            The user attempted to extrapolate when it isn't allowed.
        """
        # TODO: extrapolation checks
        res_idxs: npt.NDArray[np.int_] = (
            np.searchsorted(a=self.x, v=np.atleast_1d(x), side="left") - 1
        )
        # Fix up any overrun
        res_idxs[res_idxs == -1] = 0
        res = self.values[res_idxs]

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
        # Late import to avoid circularity
        from continuous_timeseries.timeseries_continuous import (
            ContinuousFunctionScipyPPoly,
        )

        try:
            import scipy.interpolate
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "PPolyPiecewiseConstantPreviousLeftOpen.integrate",
                requirement="scipy",
            ) from exc

        x = self.x
        x_steps = x[1:] - x[:-1]

        values_in_x_window = self.values[:-1]
        gradients = values_in_x_window
        definite_integrals = np.cumsum(values_in_x_window * x_steps)

        constant_terms = np.hstack(
            [
                integration_constant,
                # Combination of gradient and starting term ensures
                # that the last value is correct too,
                # hence why we can/must have the [:-1] indexing here.
                integration_constant + definite_integrals[:-1],
            ]
        )

        c = np.vstack([gradients, constant_terms])

        indefinite_integral = scipy.interpolate.PPoly(
            x=x,
            c=c,
            extrapolate=False,
        )

        c_new = indefinite_integral.c
        c_new[-1, :] = c_new[-1, :] - indefinite_integral(domain_start)  # type: ignore # scipy-stubs expects array

        ppoly_integral = scipy.interpolate.PPoly(
            c=c_new,
            x=indefinite_integral.x,
            extrapolate=False,
        )

        return ContinuousFunctionScipyPPoly(ppoly_integral)

    def differentiate(self) -> ContinuousFunctionScipyPPoly:
        """
        Differentiate

        Returns
        -------
        :
            Derivative of the function
        """
        # Late import to avoid circularity
        from continuous_timeseries.timeseries_continuous import (
            ContinuousFunctionScipyPPoly,
        )

        try:
            import scipy.interpolate
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "PPolyPiecewiseConstantPreviousLeftOpen.differentiate",
                requirement="scipy",
            ) from exc

        return ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(
                x=self.x,
                c=np.atleast_2d(np.zeros(self.values.size - 1)),
                extrapolate=False,  # no extrapolation by default
            )
        )


def discrete_to_continuous_piecewise_constant_previous_left_open(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise constant

    Here we use piecewise constant, previous, left-open interpolation.
    For details, see
    [the module's docstring][continuous_timeseries.discrete_to_continuous.piecewise_constant_previous_left_open].

    Parameters
    ----------
    discrete
        Discrete timeseries to convert

    Returns
    -------
    :
        Continuous version of `discrete`
        based on piecewise constant, previous, left-open interpolation.
    """  # noqa: E501
    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        TimeseriesContinuous,
    )

    time_bounds = discrete.time_axis.bounds

    all_vals = discrete.values_at_bounds.values

    piecewise_polynomial = PPolyPiecewiseConstantPreviousLeftOpen(
        x=time_bounds.m,
        values=all_vals.m,
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=time_bounds.u,
        values_units=all_vals.u,
        function=piecewise_polynomial,
        domain=(time_bounds.min(), time_bounds.max()),
    )

    return res
