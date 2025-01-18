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


# TODO: make this support TimeAxis input too
# TODO: add check that input is sorted if input is a plain numpy array
def increase_time_axis_resolution(
    time_axis: PINT_NUMPY_ARRAY, res_increase: int
) -> PINT_NUMPY_ARRAY:
    """
    Get a higher resolution time axis

    Parameters
    ----------
    time_axis
        Time axis of which to increase the resolution

    res_increase
        The increase in resolution we want.

        In each window defined by `time_axis[n]` to `time_axis[n + 1]`,
        we create `res_increase - 1` evenly spaced points
        between `time_axis[n]` and `time_axis[n + 1]`.
        The points defined by `time_axis` are also included.
        As a result, the total number of plotted points is equal to
        `time_axis.size + (res_increase - 1) * (time_axis.size - 1)`.

    Returns
    -------
    :
        Time axis with higher resolution

    Examples
    --------
    >>> import pint
    >>> UR = pint.get_application_registry()
    >>> Q = UR.Quantity
    >>>
    >>> time_axis = Q([2000, 2010, 2020, 2025], "yr")
    >>>
    >>> # Passing in res_increase equal to 1 simply returns the input values
    >>> increase_time_axis_resolution(time_axis, res_increase=1)
    <Quantity([2000. 2010. 2020. 2025.], 'year')>
    >>>
    >>> # 'Double' the resolution
    >>> increase_time_axis_resolution(time_axis, res_increase=2)
    <Quantity([2000.  2005.  2010.  2015.  2020.  2022.5 2025. ], 'year')>
    >>>
    >>> # 'Triple' the resolution
    >>> increase_time_axis_resolution(time_axis, res_increase=3)
    <Quantity([2000.         2003.33333333 2006.66666667 2010.         2013.33333333
     2016.66666667 2020.         2021.66666667 2023.33333333 2025.        ], 'year')>
    """
    time_axis_internal = time_axis[:-1]
    step_fractions = np.linspace(0.0, (res_increase - 1) / res_increase, res_increase)
    time_deltas = time_axis[1:] - time_axis[:-1]

    time_axis_rep = (
        np.repeat(time_axis_internal.m, step_fractions.size) * time_axis_internal.u
    )
    step_fractions_rep = np.tile(step_fractions, time_axis_internal.size)
    time_axis_deltas_rep = np.repeat(time_deltas.m, step_fractions.size) * time_deltas.u

    res: PINT_NUMPY_ARRAY = np.hstack(  # type: ignore # mypy confused by numpy and pint
        [
            time_axis_rep + time_axis_deltas_rep * step_fractions_rep,
            time_axis[-1],
        ]
    )

    return res
