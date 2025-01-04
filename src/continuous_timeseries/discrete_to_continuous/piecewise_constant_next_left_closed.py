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

from typing import TYPE_CHECKING, Any

import attr
import numpy as np
import numpy.typing as npt
from attrs import define, field

from continuous_timeseries.domain_helpers import check_no_times_outside_domain
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.typing import NP_ARRAY_OF_FLOAT_OR_INT, NP_FLOAT_OR_INT

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


@define
class PPolyPiecewiseConstantNextLeftClosed:
    """
    Piecewise polynomial that implements our 'next' constant left-closed logic

    We can't use [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly] directly
    because it doesn't behave as we want at the first boundary.
    We could subclass [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly],
    but that is more trouble than its worth for such a simple implementation.
    """

    x: NP_ARRAY_OF_FLOAT_OR_INT
    """
    Breakpoints between each piecewise constant interval
    """

    y: NP_ARRAY_OF_FLOAT_OR_INT = field()
    """
    The y-values which help define our spline.

    Note that these are not the same as the values at our boundaries,
    see [the module's docstring][(m)].
    """

    @y.validator
    def y_validator(
        self,
        attribute: attr.Attribute[Any],
        value: NP_ARRAY_OF_FLOAT_OR_INT,
    ) -> None:
        """
        Validate the received `y`
        """
        if value.shape != self.x.shape:
            msg = (
                "`y` and `self.x` must have the same shape. "
                f"Received: y.shape={value.shape}. {self.x.shape=}"
            )
            raise AssertionError(msg)

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
        if not allow_extrapolation:
            # If need be, no cover this, should never be used anyway
            check_no_times_outside_domain(times=x, domain=(self.x.min(), self.x.max()))

        res_idxs: npt.NDArray[np.int_] = np.searchsorted(
            a=self.x, v=np.atleast_1d(x), side="right"
        )

        # Fix up any overrun
        res_idxs[res_idxs == self.y.size] = self.y.size - 1
        res = self.y[res_idxs]

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
                "PPolyPiecewiseConstantNextLeftOpen.integrate",
                requirement="scipy",
            ) from exc

        # We have to ensure that we get the gradient outside our bounds correct,
        # in case the user wants to do extrapolation after integration.
        # Hence also consider what happens either side of the bounds.
        # We can pick points in the middle of our windows,
        # because our function is piecewise constant
        # and this helps us avoid the value at the bound headache.
        out_gradient_eval_points = np.hstack(
            [
                2 * self.x[0] - self.x[1],
                (self.x[1:] + self.x[:-1]) / 2.0,
                2 * self.x[-1] - self.x[-2],
            ]
        )
        out_gradients = self(out_gradient_eval_points, allow_extrapolation=True)

        # Grab points on either side of our domain too,
        # so that we give a correct representation,
        # irrespective of whether we're doing next or previous logic.
        x = np.hstack(
            [out_gradient_eval_points[0], self.x, out_gradient_eval_points[-1]]
        )

        change_in_windows = out_gradients * (x[1:] - x[:-1])
        tmp_constant_terms = np.hstack([0.0, np.cumsum(change_in_windows[:-1])])

        c = np.vstack([out_gradients, tmp_constant_terms])

        indefinite_integral = scipy.interpolate.PPoly(
            x=x,
            c=c,
            extrapolate=True,
        )

        c_new = indefinite_integral.c
        c_new[-1, :] = (
            c_new[-1, :] + integration_constant - indefinite_integral(domain_start)  # type: ignore # scipy-stubs expects array
        )

        ppoly_integral = scipy.interpolate.PPoly(
            c=c_new,
            x=indefinite_integral.x,
            extrapolate=True,
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
                "PPolyPiecewiseConstantNextLeftOpen.differentiate",
                requirement="scipy",
            ) from exc

        return ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(
                x=self.x,
                c=np.atleast_2d(np.zeros(self.y.size - 1)),
                extrapolate=False,  # no extrapolation by default
            )
        )


def discrete_to_continuous_piecewise_constant_next_left_closed(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise constant

    Here we use piecewise constant, next, left-closed interpolation.
    For details, see [the module's docstring][(m)].

    Parameters
    ----------
    discrete
        Discrete timeseries to convert

    Returns
    -------
    :
        Continuous version of `discrete`
        based on piecewise constant, next, left-closed interpolation.
    """
    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous

    time_bounds = discrete.time_axis.bounds

    all_vals = discrete.values_at_bounds.values

    continuous_representation = PPolyPiecewiseConstantNextLeftClosed(
        x=time_bounds.m,
        y=all_vals.m,
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=time_bounds.u,
        values_units=all_vals.u,
        function=continuous_representation,
        domain=(np.min(time_bounds), np.max(time_bounds)),
    )

    return res
