"""
Conversion of timeseries from discrete to continuous

This supports the [`TimeseriesDiscrete`][(p)] and [`TimeseriesContinuous`][(p)] APIs,
but is in more general where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


def discrete_to_continuous(  # noqa: PLR0911
    discrete: TimeseriesDiscrete, interpolation: InterpolationOption
) -> TimeseriesContinuous:
    """
    Convert a discrete timeseries to continuous

    Parameters
    ----------
    discrete
        Discrete timeseries to convert

    interpolation
        Interpolation type to use for converting from discrete to continuous.

    Returns
    -------
    :
        Continuous version of `discrete` based on `interpolation`.
    """
    if interpolation == InterpolationOption.PiecewiseConstantNextLeftClosed:
        return discrete_to_continuous_piecewise_constant_next_left_closed(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftOpen:
        return discrete_to_continuous_piecewise_constant_next_left_open(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftClosed:
        return discrete_to_continuous_piecewise_constant_previous_left_closed(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftOpen:
        return discrete_to_continuous_piecewise_constant_previous_left_open(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.Linear:
        return discrete_to_continuous_linear(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.Quadratic:
        return discrete_to_continuous_higher_order(discrete=discrete, order=2)

    if interpolation == InterpolationOption.Cubic:
        return discrete_to_continuous_higher_order(discrete=discrete, order=3)

    if interpolation == InterpolationOption.Quartic:
        return discrete_to_continuous_higher_order(discrete=discrete, order=4)

    raise NotImplementedError(interpolation.name)  # pragma: no cover


__all__ = ["InterpolationOption", "discrete_to_continuous"]
