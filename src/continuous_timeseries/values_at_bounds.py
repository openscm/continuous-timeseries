"""
Definition of [`ValuesAtBounds`][continuous_timeseries.values_at_bounds.ValuesAtBounds]

This is a container that stores a series of values.
It is designed to be compatible with the
[`TimeAxis`][continuous_timeseries.TimeAxis],
[`Timeseries`][continuous_timeseries.Timeseries]
and [`TimeseriesDiscrete`][continuous_timeseries.TimeseriesDiscrete]
classes.
The idea is that, for each time bound defined by a
[`TimeAxis`][continuous_timeseries.TimeAxis],
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

import numpy.typing as npt
import pint
from attrs import define


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
    [`InterpolationOption`][continuous_timeseries.InterpolationOption]).

    The current implementation does mean that the values at each bound are contiguous,
    i.e. it is impossible to define discontinuous values
    at the bounds of each time window.
    This is deliberate, as it significantly simplifes handling.
    """

    values: pint.facets.numpy.NumpyQuantity[npt.NDArray[Any]]
    # values: pint.registry.Quantity[npt.NDArray[Any]]
    """
    Values

    Must be one-dimensional.
    """

    # TODO: __str__, __repr__ and _repr_html_
