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
from typing import TYPE_CHECKING, Any, Protocol, cast

import attr
import numpy as np
import numpy.typing as npt
from attrs import define, field

import continuous_timeseries.formatting
from continuous_timeseries.domain_helpers import (
    check_no_times_outside_domain,
    validate_domain,
)
from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.plotting_helpers import get_plot_vals
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.typing import NP_FLOAT_OR_INT, PINT_NUMPY_ARRAY, PINT_SCALAR
from continuous_timeseries.values_at_bounds import ValuesAtBounds

if TYPE_CHECKING:
    import IPython.lib.pretty
    import matplotlib.axes
    import pint.facets.plain
    import scipy.interpolate

    from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete


class ContinuousFunctionLike(Protocol):
    """
    Protocol for classes that can be used as continuous functions
    """

    def __call__(
        self, x: npt.NDArray[NP_FLOAT_OR_INT], allow_extrapolation: bool = False
    ) -> npt.NDArray[NP_FLOAT_OR_INT]:
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

    def integrate(
        self,
        integration_constant: NP_FLOAT_OR_INT,
        domain_start: NP_FLOAT_OR_INT,
    ) -> ContinuousFunctionLike:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant

            This is required for the integral to be a definite integral.

        domain_start
            The start of the domain.

            This is required to ensure that we start at the right point
            when evaluating the definite integral.

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
        for attr_to_show in ["c", "x"]:
            attribute_rows = continuous_timeseries.formatting.add_html_attribute_row(
                attr_to_show,
                continuous_timeseries.formatting.get_html_repr_safe(
                    getattr(self.ppoly, attr_to_show)
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
    def order_str(self) -> str:
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
        self, x: npt.NDArray[NP_FLOAT_OR_INT], allow_extrapolation: bool = False
    ) -> npt.NDArray[NP_FLOAT_OR_INT]:
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
        res = cast(
            npt.NDArray[NP_FLOAT_OR_INT],
            self.ppoly(x=x, extrapolate=allow_extrapolation),
        )

        return res

    def integrate(
        self,
        integration_constant: NP_FLOAT_OR_INT,
        domain_start: NP_FLOAT_OR_INT,
    ) -> ContinuousFunctionLike:
        """
        Integrate

        Parameters
        ----------
        integration_constant
            Integration constant

            This is required for the integral to be a definite integral.

        domain_start
            The start of the domain.

            This is required to ensure that we start at the right point
            when evaluating the definite integral.

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
        c_new[-1, :] = (
            c_new[-1, :]
            + integration_constant
            - indefinite_integral(domain_start, extrapolate=True)  # type: ignore # scipy-stubs expects array
        )

        ppoly_integral = scipy.interpolate.PPoly(
            c=c_new,
            x=indefinite_integral.x,
            extrapolate=False,  # no extrapolation by default
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

    time_units: pint.facets.plain.PlainUnit
    """The units of the time axis"""

    values_units: pint.facets.plain.PlainUnit
    """The units of the values"""

    function: ContinuousFunctionLike
    """
    The continuous function that represents this timeseries.
    """

    domain: tuple[PINT_SCALAR, PINT_SCALAR] = field()
    """
    Domain over which the function can be evaluated
    """

    @domain.validator
    def domain_validator(
        self,
        attribute: attr.Attribute[Any],
        value: tuple[PINT_SCALAR, PINT_SCALAR],
    ) -> None:
        """
        Validate the received values
        """
        try:
            validate_domain(value)
        except AssertionError as exc:
            msg = "The value supplied for `domain` failed validation."
            raise ValueError(msg) from exc

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

    def to_discrete_timeseries(
        self,
        time_axis: TimeAxis,
        allow_extrapolation: bool = False,
    ) -> TimeseriesDiscrete:
        """
        Convert to [`TimeseriesDiscrete`][(p)]

        Parameters
        ----------
        time_axis
            Time axis to use for the conversion

        allow_extrapolation
            Should extrapolation be allowed during the conversion?

        Returns
        -------
        :
            Discrete representation of `self`
        """
        # Late import to avoid circularity
        from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete

        res = TimeseriesDiscrete(
            name=self.name,
            time_axis=time_axis,
            values_at_bounds=ValuesAtBounds(
                self.interpolate(time_axis, allow_extrapolation=allow_extrapolation)
            ),
        )

        return res

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

        if not allow_extrapolation:
            try:
                check_no_times_outside_domain(
                    time_axis,
                    domain=self.domain,
                )
            except ValueError as exc:
                msg = f"Extrapolation is not allowed ({allow_extrapolation=})."
                raise ExtrapolationNotAllowedError(msg) from exc

        times_m = time_axis.to(self.time_units).m
        values_m = self.function(
            times_m,
            # We have already checked the domain above.
            # Hence, we want the function to extrapolate if needed.
            allow_extrapolation=True,
        )

        if np.isnan(values_m).any():  # pragma: no cover
            # This is an escape hatch.
            # In general, we expect `self.function` to handle NaNs
            # before we get to this point.
            msg = (
                "The result of calling `self.function` contains NaNs. "
                f"The result is {values_m!r}."
            )
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
            integration_constant=integration_constant.to(integral_values_units).m,
            domain_start=self.domain[0].to(self.time_units).m,
        )

        return type(self)(
            name=name_res,
            time_units=self.time_units,
            values_units=integral_values_units,
            function=integral,
            domain=self.domain,
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
            domain=self.domain,
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

            If equal to 1, then only the points in `time_axis` will be plotted.
            If equal to 100, then there will be 100 times as many points
            plotted as the number of points in `time_axis`.
            If equal to n, then there will be n times as many points
            plotted as the number of points in `time_axis`.

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

        # Interpolate based on res_increase.
        # Then plot interpolated using linear joins
        # (as far as I can tell, this is the only general way to do this,
        # although it is slower than using e.g. step for piecewise constant stuff).)
        plot_points = get_plot_points(time_axis, res_increase=res_increase)
        plot_values = self.interpolate(plot_points)

        x_vals = get_plot_vals(
            plot_points,
            "time_axis",
            warn_if_magnitudes=warn_if_plotting_magnitudes,
        )
        y_vals = get_plot_vals(
            plot_values,
            "show_values",
            warn_if_magnitudes=warn_if_plotting_magnitudes,
        )

        ax.plot(x_vals, y_vals, label=label, **kwargs)

        return ax


def get_plot_points(time_axis: PINT_NUMPY_ARRAY, res_increase: int) -> PINT_NUMPY_ARRAY:
    """
    Get points to plot

    Parameters
    ----------
    time_axis
        Time axis to use for plotting

    res_increase
        The increase in resolution we want to use when plotting.

        In each window defined by `time_axis[n]` to `time_axis[n + 1]`,
        `res_increase - 1` evenly spaced points
        between `time_axis[n]` and `time_axis[n + 1]` will be generated.
        The points defined by `time_axis` are also included.
        As a result, the total number of plotted points is equal to
        `time_axis.size + (res_increase - 1) * (time_axis.size - 1)`.

    Returns
    -------
    :
        Points to plot

    Examples
    --------
    >>> import pint
    >>> UR = pint.get_application_registry()
    >>> Q = UR.Quantity
    >>>
    >>> time_axis = Q([2000, 2010, 2020, 2025], "yr")
    >>>
    >>> # Passing in res_increase equal to 1 simply returns the input values
    >>> get_plot_points(time_axis, res_increase=1)
    <Quantity([2000. 2010. 2020. 2025.], 'year')>
    >>>
    >>> # 'Double' the resolution
    >>> get_plot_points(time_axis, res_increase=2)
    <Quantity([2000.  2005.  2010.  2015.  2020.  2022.5 2025. ], 'year')>
    >>>
    >>> # 'Triple' the resolution
    >>> get_plot_points(time_axis, res_increase=3)
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
