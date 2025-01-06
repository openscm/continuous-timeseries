"""
Common functions used across our piecewise constant implementations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import attr
import numpy as np
import numpy.typing as npt

from continuous_timeseries.domain_helpers import check_no_times_outside_domain
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.typing import (
    NP_ARRAY_OF_FLOAT_OR_INT,
    NP_FLOAT_OR_INT,
    PINT_NUMPY_ARRAY,
)

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
        TimeseriesContinuous,
    )


class PiecewiseConstantLike(Protocol):
    """
    Piecewise-constant like implementation
    """

    x: NP_ARRAY_OF_FLOAT_OR_INT
    """
    Breakpoints between each piecewise constant interval
    """

    def __init__(
        self, x: NP_ARRAY_OF_FLOAT_OR_INT, y: NP_ARRAY_OF_FLOAT_OR_INT
    ) -> None:
        """
        Initialise

        Parameters
        ----------
        x
            Breakpoints between each piecewise constant interval

        y
            The y-values which help define our spline
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

    def differentiate(self) -> ContinuousFunctionScipyPPoly:
        """
        Differentiate

        Returns
        -------
        :
            Derivative of the function
        """

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


def piecewise_constant_y_validator(
    self: PiecewiseConstantLike,
    attribute: attr.Attribute[Any],
    value: NP_ARRAY_OF_FLOAT_OR_INT,
) -> None:
    """
    Validate the `y` attribute of piecewise constant implementations

    Parameters
    ----------
    self
        Instance of a class that is piecewise-constant like

    attribute
        Attribute being validated

    value
        Value to validate

    Raises
    ------
    AssertionError
        Value does not have the same shape as `self.x`.
    """
    if value.shape != self.x.shape:
        msg = (
            f"`{attribute.name}` and `self.x` must have the same shape. "
            f"Received: {attribute.name}.shape={value.shape}. {self.x.shape=}"
        )
        raise AssertionError(msg)


class GetIdxsCallable(Protocol):
    """
    Callable that can be used to get the index to return from `self_y`
    """

    def __call__(
        self,
        times: NP_ARRAY_OF_FLOAT_OR_INT,
        self_x: NP_ARRAY_OF_FLOAT_OR_INT,
    ) -> npt.NDArray[np.integer]:
        """
        Get the index to return from `self.y`

        Parameters
        ----------
        times
            Times for which to get values

        self_x
            `self.x`

        Returns
        -------
        :
            Indexes to return from `self.y`
        """


def get_values(
    times: NP_ARRAY_OF_FLOAT_OR_INT,
    self_x: NP_ARRAY_OF_FLOAT_OR_INT,
    self_y: NP_ARRAY_OF_FLOAT_OR_INT,
    get_idxs: GetIdxsCallable,
    allow_extrapolation: bool,
) -> NP_ARRAY_OF_FLOAT_OR_INT:
    """
    Get the values to return for a given set of times of interest

    Parameters
    ----------
    times
        Times for which to get the values

    self_x
        `self.x` i.e. the x-points which define our continuous representation

    self_y
        `self.y` i.e. the y-values which define our continuous representation

    get_idxs
        Function which defines the continuous piecewise constant's form.

        Given `times` and `self_x` returns the indexes from `self_y` to return

    allow_extrapolation
        Whether extrapolation should be allowed while retrieving the values

    Returns
    -------
    :
        Values of the function at `times`
    """
    if not allow_extrapolation:
        check_no_times_outside_domain(times=times, domain=(self_x.min(), self_x.max()))

    res_idxs = get_idxs(times=times, self_x=self_x)

    res: NP_ARRAY_OF_FLOAT_OR_INT = self_y[res_idxs]  # not sure why mypy is confused

    return res


def differentiate_piecewise_constant(
    self: PiecewiseConstantLike,
) -> ContinuousFunctionScipyPPoly:
    """
    Differentiate a piecewise-constant instance

    Parameters
    ----------
    self
        Piecewise-constant instance to differentiate

    Returns
    -------
    :
        Derivative of `self`
    """
    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
    )

    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "differentiate_piecewise_constant", requirement="scipy"
        ) from exc

    return ContinuousFunctionScipyPPoly(
        scipy.interpolate.PPoly(
            x=self.x,
            c=np.atleast_2d(np.zeros(self.x.size - 1)),
            extrapolate=False,  # no extrapolation by default
        )
    )


def integrate_piecewise_constant(
    self: PiecewiseConstantLike,
    integration_constant: NP_FLOAT_OR_INT,
    domain_start: NP_FLOAT_OR_INT,
) -> ContinuousFunctionScipyPPoly:
    """
    Integrate a piecewise-constant instance

    Parameters
    ----------
    self
        Piecewise-constant instance to integrate

    integration_constant
        Integration constant to use when calculating the integral

    domain_start
        Start of the domain of integration

    Returns
    -------
    :
        Integral of `self`
    """
    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import (
        ContinuousFunctionScipyPPoly,
    )

    try:
        import scipy.interpolate
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "integrate_piecewise_constant",
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
    x = np.hstack([out_gradient_eval_points[0], self.x, out_gradient_eval_points[-1]])

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
        extrapolate=False,  # no extrapolation by default
    )

    return ContinuousFunctionScipyPPoly(ppoly_integral)


def discrete_to_continuous_piecewise_constant(
    x: PINT_NUMPY_ARRAY,
    y: PINT_NUMPY_ARRAY,
    name: str,
    piecewise_constant_like: type[PiecewiseConstantLike],
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to piecewise constant

    Parameters
    ----------
    x
        The discrete x-values from which to convert

    y
        The discrete y-values from which to convert

    name
        The value to use to set the result's name attribute

    piecewise_constant_like
        Piecewise-constant class to convert to

    Returns
    -------
    :
        Continuous version of `x` and `y` based on `piecewise_constant_like`.
    """
    # Late import to avoid circularity
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous

    continuous_representation = piecewise_constant_like(
        x=x.m,
        y=y.m,
    )

    res = TimeseriesContinuous(
        name=name,
        time_units=x.u,
        values_units=y.u,
        function=continuous_representation,
        domain=(np.min(x), np.max(x)),
    )

    return res
