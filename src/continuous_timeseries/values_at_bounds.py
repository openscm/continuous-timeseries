"""
Definition of [`ValuesAtBounds`][(m).ValuesAtBounds]

This is a container that stores a series of values.
It is designed to be compatible with the
[`TimeAxis`][(p).TimeAxis],
[`Timeseries`][(p).Timeseries]
and [`TimeseriesDiscrete`][(p).TimeseriesDiscrete]
classes.
The idea is that, for each time bound defined by a
[`TimeAxis`][(p).TimeAxis],
it is unambiguous what value to use to use at that point in time.

As background, we considered only defining the values
that apply within each time window.
This was rejected because it only supports piecewise constant interpolation
(at least trivially).
For linear or higher order interpolation, you need to know the value
at the end of the last bound for the values to be unambiguous
over the entirety of each time window.
"""

from __future__ import annotations

from typing import Any

import attr
from attrs import define, field

from continuous_timeseries.typing import PINT_NUMPY_ARRAY


@define
class ValuesAtBounds:
    """
    Container for values to be used at the bounds of each time window in a timeseries

    This is a low-level container.
    It generally won't be used directly.

    It is important to keep in mind that this container is quite low-level.
    As a result, it does not provide all the information required to do operations,
    such as interpolation, integration and differentiation, unambiguosly.
    For example, how to interpolate between the values.
    That information has to come from other classes/information.
    For example, the kind of interpolation
    (that instead comes from
    [`InterpolationOption`][(p).InterpolationOption]).

    The current implementation does mean that the values at each bound are contiguous,
    i.e. it is impossible to define discontinuous values
    at the bounds of each time window.
    This is deliberate, as it significantly simplifes handling.
    """

    values: PINT_NUMPY_ARRAY = field()
    """
    Values

    Must be one-dimensional.
    """

    @values.validator
    def values_validator(
        self,
        attribute: attr.Attribute[Any],
        value: PINT_NUMPY_ARRAY,
    ) -> None:
        """
        Validate the received values
        """
        if len(value.shape) != 1:
            msg = (
                "`values` must be one-dimensional. "
                f"Received `values` with shape {value.shape}"
            )
            raise AssertionError(msg)

    # TODO: __str__, __repr__ and _repr_html_
