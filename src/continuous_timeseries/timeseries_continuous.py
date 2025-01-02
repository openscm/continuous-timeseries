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

import textwrap
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
from attrs import define

import continuous_timeseries.formatting
from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.plotting_helpers import get_plot_vals
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

        Raises
        ------
        ExtrapolationNotAllowedError
            The user attempted to extrapolate when it isn't allowed.

            Raising this has to be managed by the classes
            that implement this interface as only they know
            the domain over which they are defined.
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

    # Let attrs take care of __repr__

    def __str__(self) -> str:
        """
        Get string representation of self
        """
        type_self = type(self).__name__

        type_ppoly = type(self.ppoly)
        ppoly_display = f"{type_ppoly.__module__}.{type_ppoly.__name__}"

        ppoly_x = self.ppoly.x
        ppoly_c = self.ppoly.c

        order_s = self.order_str

        res = (
            f"{order_s} order {type_self}("
            f"ppoly={ppoly_display}(c={ppoly_c}, x={ppoly_x})"
            ")"
        )

        return res

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
        type_self = type(self).__name__

        type_ppoly = type(self.ppoly)
        ppoly_display = f"{type_ppoly.__module__}.{type_ppoly.__name__}"

        ppoly_x = self.ppoly.x
        ppoly_c = self.ppoly.c

        order_s = self.order_str

        with p.group(indent, f"{order_s} order {type_self}(", ")"):
            p.breakable("")  # type: ignore
            with p.group(indent, f"ppoly={ppoly_display}(", ")"):
                p.breakable("")  # type: ignore

                p.text("c=")  # type: ignore
                p.pretty(ppoly_c)  # type: ignore
                p.text(",")  # type: ignore
                p.breakable()  # type: ignore

                p.text("x=")  # type: ignore
                p.pretty(ppoly_x)  # type: ignore

    def _repr_html_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        type_self = type(self)
        header = f"{type_self.__module__}.{type_self.__name__}"

        repr_internal_row = self._repr_html_internal_row_()

        return continuous_timeseries.formatting.apply_ct_html_styling(
            display_name=header, attribute_table=repr_internal_row
        )

    def _repr_html_internal_row_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        attribute_rows: list[str] = []
        attribute_rows = continuous_timeseries.formatting.add_html_attribute_row(
            "order",
            continuous_timeseries.formatting.get_html_repr_safe(self.order),
            attribute_rows,
        )
        for attr in ["c", "x"]:
            attribute_rows = continuous_timeseries.formatting.add_html_attribute_row(
                attr,
                continuous_timeseries.formatting.get_html_repr_safe(
                    getattr(self.ppoly, attr)
                ),
                attribute_rows,
            )

        attribute_table = continuous_timeseries.formatting.make_html_attribute_table(
            attribute_rows
        )
        html_l = [
            "<table><tbody>",
            "  <tr>",
            "    <th>ppoly</th>",
            "    <td style='text-align:left;'>",
            textwrap.indent(attribute_table, "      "),
            "    </td>",
            "  </tr>",
            "</tbody></table>",
        ]

        return "\n".join(html_l)

    @property
    def order(self) -> int:
        """
        Order of the polynomial used by this instance
        """
        return self.ppoly.c.shape[0] - 1

    @property
    def order_str(self) -> int:
        """
        String name for the order of the polynomial used by this instance
        """
        order = self.order

        if order == 1:
            order_str = "1st"
        elif order == 2:  # noqa: PLR2004
            order_str = "2nd"
        elif order == 3:  # noqa: PLR2004
            order_str = "3rd"
        else:
            order_str = f"{order}th"

        return order_str

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

        Raises
        ------
        ExtrapolationNotAllowedError
            The user attempted to extrapolate when it isn't allowed.
        """
        res = self.ppoly(x=x, extrapolate=allow_extrapolation)

        if np.isnan(res).any():
            if allow_extrapolation:  # pragma: no cover
                msg = (
                    f"The result contains NaNs, even though {allow_extrapolation=}."
                    f"Result of calling `self.ppoly` was {res!r}."
                )
                raise AssertionError(msg)

            outside_x = np.hstack(
                [
                    x[np.where(x < self.ppoly.x.min())],
                    x[np.where(x > self.ppoly.x.max())],
                ]
            )
            if outside_x.size < 1:  # pragma: no cover
                # Should be impossible, but just in case
                msg = (
                    f"The result contains NaNs, "
                    "even though all the interpolation values "
                    "are within the piecewise polynomial's domain. "
                    f"{x=}. {self.ppoly.x=}. "
                    f"Result of calling `self.ppoly` was {res!r}."
                )
                raise AssertionError(msg)

            msg = (
                f"The result contains NaNs. "
                "This is because you tried to extrapolate "
                f"even though {allow_extrapolation=}. "
                f"The x-values that are outside the known domain are {outside_x}. "
                f"Result of calling `self.ppoly` was {res!r}. {self.ppoly.x=}."
            )
            raise ExtrapolationNotAllowedError(msg)

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
        try:
            values_m = self.function(times_m, allow_extrapolation=allow_extrapolation)
        except ExtrapolationNotAllowedError as exc:  # pragma: no cover
            if allow_extrapolation:
                msg = (
                    "`self.function` raised a `ExtrapolationNotAllowedError`, "
                    f"even though {allow_extrapolation=}. "
                    "Please check the implementation of `self.function`. "
                    f"{self.function=}"
                )
                raise AssertionError(msg) from exc

            raise

        if np.isnan(values_m).any():  # pragma: no cover
            # This is an escape hatch.
            # In general, we expect `self.function` to handle NaNs
            # before we get to this point.
            # (If we added a `domain` property to ContinuousFunctionLike
            # then we could simplify this.)
            msg_l = ["The result of calling `self.function` contains NaNs."]
            if not allow_extrapolation:
                msg_l.append(
                    "This might be the result of extrapolating when it is not allowed "
                    f"({allow_extrapolation=})."
                )

            msg_l.append(f"Result of calling `self.function` was {values_m!r}.")
            msg = " ".join(msg_l)
            raise AssertionError(msg)

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

        x_vals = get_plot_vals(
            show_time_points,
            "time_axis",
            warn_if_plotting_magnitudes=warn_if_plotting_magnitudes,
        )
        y_vals = get_plot_vals(
            show_values,
            "show_values",
            warn_if_plotting_magnitudes=warn_if_plotting_magnitudes,
        )

        ax.plot(x_vals, y_vals, label=label, **kwargs)

        return ax
