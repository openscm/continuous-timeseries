"""
Definition of [`ValuesAtBounds`][(m)]

This is a container that stores a series of values.
It is designed to be compatible with the
[`TimeAxis`][(p)],
[`Timeseries`][(p)]
and [`TimeseriesDiscrete`][(p)]
classes.
The idea is that, for each time bound defined by a
[`TimeAxis`][(p)],
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

from typing import TYPE_CHECKING, Any

import attr
from attrs import define, field

import continuous_timeseries.formatting
from continuous_timeseries.typing import PINT_NUMPY_ARRAY

if TYPE_CHECKING:
    import IPython.lib.pretty


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
    [`InterpolationOption`][(p)]).

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
        try:
            shape = value.shape
        except AttributeError as exc:
            msg = (
                "`values` must be one-dimensional but "
                "an error was raised while trying to check its shape. "
                f"Received values={value}."
            )
            raise AssertionError(msg) from exc

        if len(shape) != 1:
            msg = (
                "`values` must be one-dimensional. "
                f"Received `values` with shape {shape}"
            )
            raise AssertionError(msg)

    # Let attrs take care of __repr__

    def __str__(self) -> str:
        """
        Get string representation of self
        """
        return continuous_timeseries.formatting.to_str(
            self,
            [a.name for a in self.__attrs_attrs__],
        )

    def _repr_pretty_(
        self, p: IPython.lib.pretty.RepresentationPrinter, cycle: bool
    ) -> None:
        """
        Get IPython pretty representation of self

        Used by IPython notebooks and other tools
        """
        continuous_timeseries.formatting.to_pretty(
            self,
            [a.name for a in self.__attrs_attrs__],
            p=p,
            cycle=cycle,
        )

    def _repr_html_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        return continuous_timeseries.formatting.to_html(
            self, [a.name for a in self.__attrs_attrs__], prefix=f"{__name__}."
        )
