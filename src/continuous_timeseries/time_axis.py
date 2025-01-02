"""
Definition of [`TimeAxis`][(m)]

This is a container that stores our representation of a time axis.
It is designed to be compatible with the
[`Timeseries`][(p)],
[`TimeseriesContinuous`][(p)],
and [`TimeseriesDiscrete`][(p)]
classes.
The idea is that the time axis' intent is made clear
and clearly differentiated from a plain array.

The bounds provided to the [`TimeAxis`][(m)]
instance define the bounds of each time step in the time series.
These bounds are provided as one-dimensional arrays.
The first time step runs from `bounds[0]` to `bounds[1]`,
the second from `bounds[1]` to `bounds[2]`,
the third from `bounds[2]` to `bounds[3]` etc.
(the nth time step runs from `bounds[n-1]` to `bounds[n]`).
The design means that the bounds are always contiguous.
In other words, we can have a time axis concept and API
without the headaches of having to handle arbitrary time steps,
particularly those that have gaps.
This is clearly a design trade-off.

One other consequence of this container's structure is
that you can't have bounds which start before the first value.
In other words, the start of the first timestep
is always equal to the first value held by the [`TimeAxis`][(m)] instance.
However, we can't think of a situation in which that is a problem
(and setting it up this way makes life much simpler).

As background, we considered supporting arbitrary bounds
following something like the [CF-Conventions](https://cfconventions.org/)
and [cf-python](https://github.com/NCAS-CMS/cf-python).
However, this introduces many more complexities.
For our use case, these were not deemed desirable or relevant
so we went for this simplified approach.
If there were a clear use case,
it would probably not be difficult to create a translation between
this package and e.g. [cf-python](https://github.com/NCAS-CMS/cf-python).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attr
import numpy as np
from attrs import define, field

import continuous_timeseries.formatting
from continuous_timeseries.typing import PINT_NUMPY_ARRAY

if TYPE_CHECKING:
    import IPython.lib.pretty


@define
class TimeAxis:
    """
    Time axis representation
    """

    bounds: PINT_NUMPY_ARRAY = field()
    """
    Bounds of each time step in the time axis.

    Must be one-dimensional and monotonically increasing.

    The first time step runs from `bounds[0]` to `bounds[1]`,
    the second from `bounds[1]` to `bounds[2]`,
    the third from `bounds[2]` to `bounds[3]` etc.
    (the nth step runs from `bounds[n-1]` to `bounds[n]`).

    As a result, if `bounds` has length n, then it defines n - 1 time steps.
    """

    @bounds.validator
    def bounds_validator(
        self,
        attribute: attr.Attribute[Any],
        value: PINT_NUMPY_ARRAY,
    ) -> None:
        """
        Validate the received bounds
        """
        try:
            shape = value.shape
        except AttributeError as exc:
            msg = (
                "`bounds` must be one-dimensional but "
                "an error was raised while trying to check its shape. "
                f"Received bounds={value}."
            )
            raise AssertionError(msg) from exc

        if len(shape) != 1:
            msg = (
                "`bounds` must be one-dimensional. "
                f"Received `bounds` with shape {shape}"
            )
            raise AssertionError(msg)

        deltas = value[1:] - value[:-1]
        if (deltas <= 0).any():
            msg = (
                "`bounds` must be strictly monotonically increasing. "
                f"Received bounds={value}"
            )
            raise ValueError(msg)

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
        self,
        p: IPython.lib.pretty.RepresentationPrinter,
        cycle: bool,
        indent: int = 4,
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

    def _repr_html_internal_row_(self) -> str:
        """
        Get html representation of self to use as an internal row of another object

        Used to avoid our representations having more information than we'd like.
        """
        return continuous_timeseries.formatting.to_html(
            self,
            [a.name for a in self.__attrs_attrs__],
            include_header=False,
        )

    @property
    def bounds_2d(self) -> PINT_NUMPY_ARRAY:
        """
        Get the bounds of the time steps in two-dimensions

        This representation can be useful for some operations.

        Returns
        -------
        :
            Bounds of the time steps in two-dimensions
            (bounds is the second dimension i.e. has size 2).
        """
        starts = self.bounds[:-1]
        ends = self.bounds[1:]

        res: PINT_NUMPY_ARRAY = np.vstack([starts, ends]).T  # type: ignore # mypy confused by pint

        return res
