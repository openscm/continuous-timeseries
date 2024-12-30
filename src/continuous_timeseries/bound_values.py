"""
Definition of [`BoundValues`][continuous_timeseries.bound_values.BoundValues]

This is a container that stores a series of values.
It is designed to be compatible with the
[`TimeAxis`][continuous_timeseries.time_axis.TimeAxis] class.
The idea is that, for each value in a
[`TimeAxis`][continuous_timeseries.time_axis.TimeAxis]
instance, it is unambiguous what value to use.
In addition, this API also makes clear
that the data is being used to define values at the bounds of time windows.
As a result, we require the specification of the value at the last bound too,
an extra value compared to how people normally think about things.

As background, here are a couple of alternatives that we considered:

1. Don't have the extra data point i.e. the value at the last bound.
   Simply assume that each value applies to the window defined
   by the time bounds either side of it.

    - This was rejected because it only supports piecewise constant interpolation.
      For linear or higher order interpolation, you need to know the value
      at the end of the last bound for the values to be unambiguous
      over the entirety of each time window.

1. Just treat all values as bound values, don't distinguish `value_last_bound`.

    - This was rejected because it is the status quo.
      In our opinion, this way of treating things is not explicit enough
      to avoid developers and users shooting themselves in the foot.
      The key to this whole exercise is to force people
      to think about the choices they are making
      and what it takes to be unambiguous.
      We want an API that makes these choices as clear as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

if TYPE_CHECKING:
    import pint


@define
class BoundValues:
    """
    Container for values to be used at the bounds of each time window in a timeseries

    This is a low-level container.
    It generally won't be used directly.

    The values at the bound of each time window are defined by the values,
    with the `last_bound_value` attribute
    resolving any ambiguity about what value to use at the last bound.

    It is important to keep in mind that this container is quite low-level.
    As a result, it does not provide all the information required to do operations,
    such as interpolation, integration and differentiation, unambiguosly.
    For example, how to interpolate between the values.
    That information has to come from other classes/information.
    For example, the kind of interpolation
    (that instead comes from
    [`InterpolationOption`][continuous_timeseries.interpolation_option.InterpolationOption].
    ).

    The current implementation does mean that the values at each bound are contiguous.
    This is deliberate, as it significantly simplifes handling.
    This is clearly a design trade-off,
    although one we think could be undone later if needed.
    """

    values: pint.UnitRegistry.Quantity  # array
    """
    Values
    """

    value_last_bound: pint.UnitRegistry.Quantity  # scalar
    """
    Value to use for the last value to use in the bounds array

    Required to avoid ambiguity in this value
    """

    # TODO: __str__, __repr__ and _repr_html_

    @property
    def all_values(self) -> pint.UnitRegistry.Quantity:  # array
        """
        Get the values, including `self.value_last_bound`

        Returns
        -------
        :
            Values, including `self.value_last_bound`
        """
        res: pint.UnitRegistry.Quantity = np.hstack(  # type: ignore # mypy confused by pint overloading
            [self.values, self.value_last_bound]
        )

        return res

    @classmethod
    def from_all_values(
        cls,
        all_values: pint.UnitRegistry.Quantity,  # array
    ) -> BoundValues:
        """
        Initialise from all values

        Parameters
        ----------
        all_values
            Values with which to initialise the class.

            We assume that `all_values[-1]` is to be used for `self.value_last_bound`.

        Returns
        -------
        :
            Initialised instance
        """
        return cls(
            values=all_values[:-1],
            value_last_bound=all_values[-1],
        )
