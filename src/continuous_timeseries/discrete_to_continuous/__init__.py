"""
Conversion of timeseries from discrete to continuous

This supports the [`TimeseriesDiscrete`][(p)] and [`TimeseriesContinuous`][(p)] APIs,
but is in more general where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from continuous_timeseries.typing import PINT_NUMPY_ARRAY

from .higher_order import discrete_to_continuous_higher_order
from .interpolation_option import (
    InterpolationOption,
)
from .linear import discrete_to_continuous_linear
from .piecewise_constant_next_left_closed import (
    discrete_to_continuous_piecewise_constant_next_left_closed,
)
from .piecewise_constant_next_left_open import (
    discrete_to_continuous_piecewise_constant_next_left_open,
)
from .piecewise_constant_previous_left_closed import (
    discrete_to_continuous_piecewise_constant_previous_left_closed,
)
from .piecewise_constant_previous_left_open import (
    discrete_to_continuous_piecewise_constant_previous_left_open,
)

if TYPE_CHECKING:
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous


def discrete_to_continuous(  # noqa: PLR0911
    x: PINT_NUMPY_ARRAY,
    y: PINT_NUMPY_ARRAY,
    interpolation: InterpolationOption,
    name: str,
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to continuous

    Parameters
    ----------
    x
        The discrete x-values from which to convert

    y
        The discrete y-values from which to convert

    interpolation
        Interpolation type to use for converting from discrete to continuous.

    name
        The value to use to set the result's name attribute

    Returns
    -------
    :
        Continuous version of `discrete` based on `interpolation`.
    """
    if interpolation == InterpolationOption.PiecewiseConstantNextLeftClosed:
        return discrete_to_continuous_piecewise_constant_next_left_closed(
            x=x,
            y=y,
            name=name,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftOpen:
        return discrete_to_continuous_piecewise_constant_next_left_open(
            x=x,
            y=y,
            name=name,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftClosed:
        return discrete_to_continuous_piecewise_constant_previous_left_closed(
            x=x,
            y=y,
            name=name,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftOpen:
        return discrete_to_continuous_piecewise_constant_previous_left_open(
            x=x,
            y=y,
            name=name,
        )

    if interpolation == InterpolationOption.Linear:
        return discrete_to_continuous_linear(
            x=x,
            y=y,
            name=name,
        )

    if interpolation == InterpolationOption.Quadratic:
        return discrete_to_continuous_higher_order(x=x, y=y, name=name, order=2)

    if interpolation == InterpolationOption.Cubic:
        return discrete_to_continuous_higher_order(x=x, y=y, name=name, order=3)

    if interpolation == InterpolationOption.Quartic:
        return discrete_to_continuous_higher_order(x=x, y=y, name=name, order=4)

    raise NotImplementedError(interpolation.name)  # pragma: no cover


__all__ = ["InterpolationOption", "discrete_to_continuous"]
