"""
Definition of a discrete timeseries ([`TimeseriesDiscrete`][(m)])

This class defines our representation of discrete time series.
It is designed to be compatible with the
[`Timeseries`][(p)]
and [`TimeseriesContinuous`][(p)].
classes.
The idea is that we have a simple, lightweight container
for handling discrete timeseries (what most people are used to).
However, there are then relatively straight-forward methods
for converting to continuous views i.e. [`TimeseriesContinuous`][(p)].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attr
from attrs import define, field

import continuous_timeseries.formatting
from continuous_timeseries.discrete_to_continuous import discrete_to_continuous
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.plotting_helpers import get_plot_vals
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.values_at_bounds import ValuesAtBounds

if TYPE_CHECKING:
    import IPython.lib.pretty
    import matplotlib.axes

    from continuous_timeseries.discrete_to_continuous.interpolation_option import (
        InterpolationOption,
    )
    from continuous_timeseries.timeseries_continuous import TimeseriesContinuous


@define
class TimeseriesDiscrete:
    """
    Discrete time series representation
    """

    name: str
    """Name of the timeseries"""

    time_axis: TimeAxis
    """Time axis of the timeseries"""

    values_at_bounds: ValuesAtBounds = field()
    """
    Values at the bounds defined by `self.time_axis`

    Must hold values that are the same length as `self.time_axis`.
    """

    @values_at_bounds.validator
    def values_at_bounds_validator(
        self,
        attribute: attr.Attribute[Any],
        value: ValuesAtBounds,
    ) -> None:
        """
        Validate the received values
        """
        if value.values.shape != self.time_axis.bounds.shape:
            msg = (
                "`values_at_bounds` must have values "
                "that are the same shape as `self.time_axis.bounds`. "
                f"Received values_at_bounds.values.shape={value.values.shape} "
                f"while {self.time_axis.bounds.shape=}."
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
            self,
            [a.name for a in self.__attrs_attrs__],
            prefix="continuous_timeseries.",
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

    def to_continuous_timeseries(
        self,
        interpolation: InterpolationOption,
    ) -> TimeseriesContinuous:
        """
        Convert to [`TimeseriesContinuous`][(p)]

        Parameters
        ----------
        interpolation
            Interpolation to use for the conversion

        Returns
        -------
        :
            Continuous representation of `self` for the interpolation
            specified by `interpolation`
        """
        return discrete_to_continuous(
            discrete=self,
            interpolation=interpolation,
        )

    def plot(
        self,
        label: str | None = None,
        ax: matplotlib.axes.Axes | None = None,
        warn_if_plotting_magnitudes: bool = True,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Plot the data

        Parameters
        ----------
        label
            Label to use when plotting the data.

            If not supplied, we use the `self.name`.

        ax
            Axes on which to plot.

            If not supplied, a set of axes will be created.

        warn_if_plotting_magnitudes
            Should a warning be raised if the units of the values
            are not considered while plotting?

        **kwargs
            Keyword arguments to pass to `ax.scatter`.

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        if label is None:
            label = self.name

        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "TimeseriesDiscrete.plot", requirement="matplotlib"
                ) from exc

            _, ax = plt.subplots()

        x_vals = get_plot_vals(
            self.time_axis.bounds,
            "self.time_axis.bounds",
            warn_if_magnitudes=warn_if_plotting_magnitudes,
        )
        y_vals = get_plot_vals(
            self.values_at_bounds.values,
            "self.values_at_bounds.values",
            warn_if_magnitudes=warn_if_plotting_magnitudes,
        )

        ax.scatter(x_vals, y_vals, label=label, **kwargs)

        return ax
