"""
Definition of a timeseries ([`Timeseries`][(m)])

This class defines our representation of time series.
This is intended to be our key user-facing class,
with [`TimeseriesContinuous`][(p)] and [`TimeseriesDiscrete`][(p)]
being more low-level.
The idea is that we have a units-aware,
operation-aware (e.g. integration and differentiation)
container for handling timeseries.
We include straight-forward methods to convert to
[`TimeseriesDiscrete`][(p)] as this is what most people are more used to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from attrs import define

import continuous_timeseries.formatting
from continuous_timeseries.discrete_to_continuous import (
    InterpolationOption,
    discrete_to_continuous,
)
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR
from continuous_timeseries.values_at_bounds import ValuesAtBounds

if TYPE_CHECKING:
    import IPython.lib.pretty
    import matplotlib.axes


@define
class Timeseries:
    """Timeseries representation"""

    time_axis: TimeAxis
    """
    Time axis of the timeseries

    Used for plotting and creating the discrete form of the time series.
    """

    timeseries_continuous: TimeseriesContinuous
    """Continuous version of the timeseries"""

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

    @property
    def name(self) -> str:
        """
        Name of the time series
        """
        return self.timeseries_continuous.name

    @property
    def discrete(self) -> TimeseriesDiscrete:
        """
        Discrete view of the time series
        """
        values_at_bounds = ValuesAtBounds(
            self.timeseries_continuous.interpolate(self.time_axis)
        )

        return TimeseriesDiscrete(
            name=self.name,
            time_axis=self.time_axis,
            values_at_bounds=values_at_bounds,
        )

    @classmethod
    def from_arrays(
        cls,
        time_axis_bounds: PINT_NUMPY_ARRAY,
        values_at_bounds: PINT_NUMPY_ARRAY,
        interpolation: InterpolationOption,
        name: str,
    ) -> Timeseries:
        """
        Initialise from arrays

        Parameters
        ----------
        time_axis_bounds
            The bounds that define the time series' time axis

        values_at_bounds
            Values that apply at the bounds
            of each time point in `time_axis_bounds`

        interpolation
            Interpolation to apply when converting
            the discrete values to a continuous representation

        name
            Name of the time series

        Returns
        -------
        :
            Initialised [`Timeseries`][(m)].
        """
        time_axis_obj = TimeAxis(time_axis_bounds)
        # Initialise from discrete arrays
        discrete = TimeseriesDiscrete(
            name=name,
            time_axis=time_axis_obj,
            values_at_bounds=ValuesAtBounds(values_at_bounds),
        )
        continuous = discrete_to_continuous(
            discrete=discrete,
            interpolation=interpolation,
        )

        return cls(
            time_axis=time_axis_obj,
            timeseries_continuous=continuous,
        )

    def differentiate(
        self,
        name_res: str | None = None,
    ) -> Timeseries:
        """
        Differentiate the time series

        Parameters
        ----------
        name_res
            Name to apply to the result.

            If not supplied, we use `f"{self.name}_derivative`.

        Returns
        -------
        :
            Derivative of the time series
        """
        if name_res is None:
            name_res = f"{self.name}_derivative"

        derivative = self.timeseries_continuous.differentiate(
            name_res=name_res,
        )

        return type(self)(
            time_axis=self.time_axis,
            timeseries_continuous=derivative,
        )

    def integrate(
        self,
        integration_constant: PINT_SCALAR,
        name_res: str | None = None,
    ) -> Timeseries:
        """
        Integrate the time series

        Parameters
        ----------
        integration_constant
            The integration constant to use when performing the integration.

            This is required to ensure that the integral is a definite integral.

        name_res
            Name to apply to the result.

            If not supplied, we use `f"{self.name}_integral`.

        Returns
        -------
        :
            Integral of the time series
        """
        if name_res is None:
            name_res = f"{self.name}_integral"

        integral = self.timeseries_continuous.integrate(
            integration_constant=integration_constant,
            name_res=name_res,
        )

        return type(self)(
            time_axis=self.time_axis,
            timeseries_continuous=integral,
        )

    def interpolate(
        self, time_axis: TimeAxis | PINT_NUMPY_ARRAY, allow_extrapolation: bool = False
    ) -> Timeseries:
        """
        Interpolate onto a new time axis

        Parameters
        ----------
        time_axis
            Time axis to update to

        allow_extrapolation
            Should extrapolation be allowed?

        Returns
        -------
        :
            `self`, interpolated onto `time_axis`.
        """
        if allow_extrapolation:
            raise NotImplementedError(allow_extrapolation)
        # TODO: add checks about extrapolation here
        if not isinstance(time_axis, TimeAxis):
            time_axis = TimeAxis(time_axis)

        return type(self)(
            time_axis=time_axis,
            timeseries_continuous=self.timeseries_continuous,
        )

    def update_interpolation(self, interpolation: InterpolationOption) -> Timeseries:
        """
        Update the interpolation

        Parameters
        ----------
        interpolation
            Interpolation to change to

        Returns
        -------
        :
            `self` with its interpolation updated to `interpolation`.
        """
        # Put note in here that using this with piecewise constant can be confusing
        # TODO: tests of what happens
        continuous = discrete_to_continuous(
            discrete=self.discrete,
            interpolation=interpolation,
        )

        return type(self)(
            time_axis=self.time_axis,
            timeseries_continuous=continuous,
        )

    def update_interpolation_integral_preserving(
        self,
        interpolation: InterpolationOption,
        name_res: str | None = None,
    ) -> Timeseries:
        """
        Update the interpolation while preserving the integral

        This is useful if the integral of your quantity needs to be preserved,
        e.g. you want to do integral-preserving interpolation of emissions
        so that mass is conserved.

        It is obviously not possible to do this at all time points.
        So it would be more precise to say
        that the integral is preserved at the points in `self.time_axis`.

        We recommend being a bit careful with this.
        In general, performing this operation with linear or higher-order interpolations
        may not lead to the most intuitive result because of the
        quadratic or higher-order fitting that is done in cumulative space
        (quadratic and higher-order fitting is a difficult problem in general,
        see, for example, the multiple boundary condition options in
        [scipy's cubic spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
        ).

        Parameters
        ----------
        interpolation
            Interpolation to update to

        name_res
            Name of the result

            If not supplied, we use
            `f"{self.name}_integral-preserving-interpolation"`.

        Returns
        -------
        :
            `self` with interpolation updated to `interpolation`
            while preserving the integral at `self.time_axis`.
        """
        if name_res is None:
            name_res = f"{self.name}_integral-preserving-interpolation"

        # Value doesn't matter as the value will be lost when we differentiate.
        integration_constant = 0.0 * (
            self.timeseries_continuous.values_units
            * self.timeseries_continuous.time_units
        )

        if interpolation in (
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftOpen,
        ):
            interpolation_cumulative = InterpolationOption.Linear

        elif interpolation in (InterpolationOption.Linear,):
            interpolation_cumulative = InterpolationOption.Quadratic

        elif interpolation in (InterpolationOption.Quadratic,):
            interpolation_cumulative = InterpolationOption.Cubic

        elif interpolation in (InterpolationOption.Cubic,):
            interpolation_cumulative = InterpolationOption.Quartic

        else:
            raise NotImplementedError(interpolation)

        res = (
            self.integrate(integration_constant)
            .update_interpolation(interpolation_cumulative)
            .differentiate(name_res=name_res)
        )

        return res

    def update_time(
        self, time_axis: TimeAxis, allow_extrapolation: bool = False
    ) -> Timeseries:
        """
        Update the time axis of `self`

        This is really just an alias for [`interpolate`][(c)].

        Parameters
        ----------
        time_axis
            Time axis to update to

        allow_extrapolation
            Should extrapolation be allowed while updating `self.time_axis`?

        Returns
        -------
        :
            `self` updated to the time axis `time_axis`
        """
        return self.interpolate(
            time_axis=time_axis, allow_extrapolation=allow_extrapolation
        )

    def plot(
        self,
        show_continuous: bool = True,
        continuous_plot_kwargs: dict[str, Any] | None = None,
        show_discrete: bool = False,
        discrete_plot_kwargs: dict[str, Any] | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot

        Parameters
        ----------
        show_continuous
            Should we plot the continuous representation of `self`?

        continuous_plot_kwargs
            Passed to `self.timeseries_continuous.plot`

            For docs, see
            [`TimeseriesContinuous.plot`][(p)].

        show_discrete
            Should we plot the discrete representation of `self`?

        discrete_plot_kwargs
            Passed to `self.timeseries_discrete.plot`

            For docs, see
            [`TimeseriesDiscrete.plot`][(p)].

        ax
            Axes on which to plot.

            If not supplied, a set of axes will be created.

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "TimeseriesContinuous.plot", requirement="matplotlib"
                ) from exc

            _, ax = plt.subplots()

        if continuous_plot_kwargs is None:
            continuous_plot_kwargs = {}

        if discrete_plot_kwargs is None:
            discrete_plot_kwargs = {}

        if show_continuous:
            self.timeseries_continuous.plot(
                time_axis=self.time_axis,
                ax=ax,
                **continuous_plot_kwargs,
            )

        if show_discrete:
            self.discrete.plot(
                ax=ax,
                **discrete_plot_kwargs,
            )

        return ax
