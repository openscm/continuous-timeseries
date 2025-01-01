"""
Definition of a continuous timeseries ([`TimeseriesContinuous`][(m)])

This class defines our representation of continuous time series.
It is designed to be compatible with the
[`Timeseries`][(p)]
and [`TimeseriesDiscrete`][(p)].
classes.
The idea is that we have a units-aware container
for handling continuous timeseries.
This allows us to implement interpolation,
integration and differentiation in a relatively trivial way.
We include straight-forward methods to convert to
[`TimeseriesDiscrete`][(p)] as this is what most people are more used to.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
from attrs import define

import continuous_timeseries.formatting
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR

if TYPE_CHECKING:
    import IPython.lib.pretty
    import matplotlib.axes
    import pint.registry
    import scipy.interpolate


class ContinuousFunctionLike(Protocol):
    """
    Protocol for classes that can be used as continuous functions
    """

    def __call__(
        self, x: npt.NDArray[np.number[Any]], allow_extrapolation: bool = False
    ) -> npt.NDArray[np.number[Any]]:
        """
        Evaluate the function at specific points

        Parameters
        ----------
        x
            Points at which to evaluate the function

        allow_extrapolation
            Should extrapolatino be allowed?

        Returns
        -------
        :
            The function, evaluated at `x`
        """

    def integrate(self, integration_constant: np.number[Any]) -> ContinuousFunctionLike:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant

            This is required for the integral to be a definite integral.

        Returns
        -------
        :
            Integral of the function
        """

    def differentiate(self) -> ContinuousFunctionLike:
        """
        Differentiate

        Returns
        -------
        :
            Derivative of the function
        """


@define
class ContinuousFunctionScipyPPoly:
    """
    Wrapper around scipy's piecewise polynomial

    The wrapper makes [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly]
    compatible with the interface expected by
    [`ContinuousFunctionLike`][(m)].
    """

    ppoly: scipy.interpolate.PPoly
    """
    Wrapped [`scipy.interpolate.PPoly`][scipy.interpolate.PPoly] instance
    """

    # TODO __str__ method

    def __call__(
        self, x: npt.NDArray[np.number[Any]], allow_extrapolation: bool = False
    ) -> npt.NDArray[np.number[Any]]:
        """
        Evaluate the function at specific points

        Parameters
        ----------
        x
            Points at which to evaluate the function

        allow_extrapolation
            Should extrapolatino be allowed?

        Returns
        -------
        :
            The function, evaluated at `x`
        """
        res = self.ppoly(x=x, extrapolate=allow_extrapolation)

        if np.isnan(res).any():
            msg = (
                f"The result contains NaNs. "
                "Was this because you tried to extrapolate "
                f"when it is not allowed ({allow_extrapolation=})?. "
                f"Result of calling `self.ppoly` was {res!r}."
            )
            raise ValueError(msg)

        return res

    def integrate(self, integration_constant: np.number[Any]) -> ContinuousFunctionLike:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant

            This is required for the integral to be a definite integral.

        Returns
        -------
        :
            Integral of the function
        """
        try:
            import scipy.interpolate
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "ContinuousFunctionScipyPPoly.integrate", requirement="scipy"
            ) from exc

        indefinite_integral = self.ppoly.antiderivative()

        c_new = indefinite_integral.c
        c_new[-1, :] = c_new[-1, :] + integration_constant

        ppoly_integral = scipy.interpolate.PPoly(
            c=c_new,
            x=indefinite_integral.x,
            extrapolate=False,
        )

        return type(self)(ppoly_integral)

    def differentiate(self) -> ContinuousFunctionLike:
        """
        Differentiate

        Returns
        -------
        :
            Derivative of the function
        """
        return type(self)(self.ppoly.derivative())


@define
class TimeseriesContinuous:
    """
    Continuous time series representation
    """

    name: str
    """Name of the timeseries"""

    time_units: pint.registry.Unit
    """The units of the time axis"""

    values_units: pint.registry.Unit
    """The units of the values"""

    function: ContinuousFunctionLike
    """
    The continuous function that represents this timeseries.
    """

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

    # # When we have discrete_to_continuous, add
    # def to_discrete_timeseries(
    #     self,
    #     time_axis: TimeAxis,
    # ) -> TimeseriesDiscrete:
    #     interpolate onto time_axis values,
    #     then return TimeseriesDiscrete with same name.

    def interpolate(
        self, time_axis: TimeAxis | PINT_NUMPY_ARRAY, allow_extrapolation: bool = False
    ) -> PINT_NUMPY_ARRAY:
        """
        Interpolate values on a given time axis

        Parameters
        ----------
        time_axis
            Time axis onto which to interpolate values

        allow_extrapolation
            Should extrapolation be allowed while interpolating?

        Returns
        -------
        :
            Interpolated values
        """
        if isinstance(time_axis, TimeAxis):
            time_axis = time_axis.bounds

        times_m = time_axis.to(self.time_units).m
        values_m = self.function(times_m, allow_extrapolation=allow_extrapolation)

        if np.isnan(values_m).any():
            msg = (
                f"The result of calling `self.function` contains NaNs. "
                f"Extrapolation when not allowed ({allow_extrapolation=})?. "
                f"Result of calling `self.function` was {values_m!r}."
            )
            raise ValueError(msg)

        res: PINT_NUMPY_ARRAY = values_m * self.values_units

        return res

    def integrate(
        self, integration_constant: PINT_SCALAR, name_res: str | None = None
    ) -> TimeseriesContinuous:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant to use when performing the integration

        name_res
            Name to use for the output.

            If not supplied, we use f"{self.name}_integral".

        Returns
        -------
        :
            Integral of `self`.
        """
        if name_res is None:
            name_res = f"{self.name}_integral"

        integral_values_units = self.values_units * self.time_units

        integral = self.function.integrate(
            integration_constant=integration_constant.to(integral_values_units).m
        )
        # indefinite_integral = self.piecewise_polynomial.antiderivative()
        #
        # c_new = indefinite_integral.c
        # c_new[-1, :] += integration_constant.to(integral_values_units).m
        #
        # # TODO: introduce wrapper class to help clean this interface up
        # # to make writing the Protocol easier.
        # piecewise_polynomial_integral = scipy.interpolate.PPoly(
        #     c=c_new,
        #     x=indefinite_integral.x,
        #     extrapolate=False,
        # )

        return type(self)(
            name=name_res,
            time_units=self.time_units,
            values_units=integral_values_units,
            function=integral,
        )

    def differentiate(self, name_res: str | None = None) -> TimeseriesContinuous:
        """
        Differentiate

        Parameters
        ----------
        name_res
            Name to use for the output.

            If not supplied, we use f"{self.name}_derivative".

        Returns
        -------
        :
            Integral of `self`.
        """
        if name_res is None:
            name_res = f"{self.name}_derivative"

        derivative_values_units = self.values_units / self.time_units

        derivative = self.function.differentiate()

        return type(self)(
            name=name_res,
            time_units=self.time_units,
            values_units=derivative_values_units,
            function=derivative,
        )

    def plot(
        self,
        time_axis: TimeAxis | PINT_NUMPY_ARRAY,
        res_increase: int = 500,
        label: str | None = None,
        ax: matplotlib.axes.Axes | None = None,
        warn_if_plotting_magnitudes: bool = True,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Plot the function

        We can't see an easy way to plot the continuous function exactly,
        so we approximate by interpolating very finely
        then just using a standard linear interpolation between the points.

        Parameters
        ----------
        time_axis
            Time axis to use for plotting.

            All points in `time_axis` will be included as plotting points.

        res_increase
            The amount by which to increase the resolution of the x-axis when plotting.

            (Note, these docs are actually incorrect.
            The algorithm needs to be updated
            to handle uneven spacing in time_axis for it to be true.)
            If equal to 1, then only the points in `time_axis` will be plotted.
            If equal to 100, then there will be roughly 100 times as many points
            plotted as the number of points in `time_axis`
            (roughly because there can be slightly fewer if there are duplicate points
            in `time_axis` and the created plotting points).

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
            Keyword arguments to pass to `ax.plot`.

        Returns
        -------
        :
            Axes on which the data was plotted
        """
        if isinstance(time_axis, TimeAxis):
            time_axis = time_axis.bounds

        if label is None:
            label = self.name

        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "TimeseriesContinuous.plot", requirement="matplotlib"
                ) from exc

            _, ax = plt.subplots()

        # Interpolate.
        # Then plot interpolated using linear joins
        # (as far as I can tell, this is the only general way to do this,
        # although it is slower than using e.g. step for piecewise constant stuff).)
        show_time_points = (
            np.union1d(
                np.linspace(
                    time_axis[0].m,
                    time_axis[-1].m,
                    time_axis.size * res_increase,
                ),
                time_axis.m,
            )
        ) * time_axis.u
        show_values = self.interpolate(show_time_points)

        def get_plot_vals(
            pint_q: PINT_NUMPY_ARRAY, desc: str
        ) -> PINT_NUMPY_ARRAY | npt.NDArray[np.number[Any]]:
            try:
                import matplotlib.units

                units_registered_with_matplotlib = (
                    type(pint_q) in matplotlib.units.registry
                )

            except ImportError:
                msg = (
                    "Could not import `matplotlib.units` "
                    "to set up unit-aware plotting. "
                    "We will simply try plotting magnitudes instead."
                )
                warnings.warn(msg, stacklevel=3)

                return pint_q.m

            if units_registered_with_matplotlib:
                return pint_q

            if warn_if_plotting_magnitudes:
                msg = (
                    f"The units of `{desc}` are not registered with matplotlib. "
                    "The magnitude will be plotted "
                    "without any consideration of units. "
                    "For docs on how to set up unit-aware plotting, see "
                    "[the stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html) "  # noqa: E501
                    "(at the time of writing, the latest version's docs were "
                    "[v0.24.4](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))."
                )
                warnings.warn(msg, stacklevel=3)

            return pint_q.m

        x_vals = get_plot_vals(show_time_points, "time_axis")
        y_vals = get_plot_vals(show_values, "show_values")

        ax.plot(x_vals, y_vals, label=label, **kwargs)

        return ax
