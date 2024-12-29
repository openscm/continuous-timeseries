# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Why this API?
#
# Here we explain the motivation for the API.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import scipy.interpolate
from attrs import evolve

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %% [markdown]
# ## Introducing the problem

# %% [markdown]
# Imagine you have timeseries of emissions, like the below.

# %%
emissions = Q(np.array([0, 1, 3, 5, 10, 10, 9, 7, 6]), "GtC / yr")
years = Q(np.array([1850, 1900, 1950, 2000, 2010, 2020, 2030, 2040, 2050]), "yr")

# %% [markdown]
# At first glance, this seems quite straightforward.
# However, there are quite a few unanswered questions with such data.
# For example:
#
# - do these values represent the average emissions for each year?
# - should they be interpolated linearly?

# %% [markdown]
# If we take a more complicated example,
# like emissions over the COVID period (around 2020),
# the issue becomes clearer.

# %%
emissions_covid = Q(np.array([10.2, 10.3, 9.5, 10.1, 10.3, 10.5]), "GtC / yr")
years_covid = Q(np.array([2018, 2019, 2020, 2021, 2022, 2023]), "yr")

# %% [markdown]
# If we naively plot these emissions, the problem is clearer.

# %%
fig, ax = plt.subplots()
ax.plot(years_covid, emissions_covid)
ax.grid()

# %% [markdown]
# When you first look at this plot, everything seems fine.
# However, if you look more closely, you realise something:
# the way this data is plotted,
# it looks like emissions started to drop sharply in 2019, not 2020.
# Put another way, this makes it look like the COVID dip was centred
# around 1 Jan 2020, when we know it is centred more around July 2020.
#
# Given we know that emissions data is generally the average of the emissions
# over the year, we can make this clearer with our plot.
# For example, the plotting below.

# %%
fig, ax = plt.subplots()
x_vals = np.hstack([years_covid, years_covid[-1] + Q(1, "yr")])
y_vals = np.hstack([emissions_covid, emissions_covid[-1]])
ax.step(x_vals, y_vals, where="post")
ax.grid()

# %% [markdown]
# As you can see, this is a bit of mucking around and we are currently just assuming
# that having a constant value over the year is the right choice,
# rather than actually knowing that to be the case.

# %% [markdown]
# The last key motivating issue is the question of integration.
# Recall the points we have.

# %%
years

# %%
emissions

# %% [markdown]
# It is clear that we have to consider the size of the timesteps in order to integrate the emissions.
# So, we want an API that makes that easy.
#
# On top of this, the decision about whether to linearly interpolate between the emissions values
# or treat them as stepwise constant (i.e. assume that emissions are constant between the defining points)
# will have a big difference on the result, yet we do not have any information about what choice was intended based on the data.
# So, we want an API that solves this too.

# %% [markdown]
# ## The proposed solution

# %% [markdown]
# Our proposed API to solve this is the below

# %%
from collections.abc import Iterable
from typing import Any


def get_attribute_str_value(instance: Any, attribute: str) -> str:
    """
    Get the string version of an attribute's value

    Parameters
    ----------
    instance
        Instance from which to get the attribute

    attribute
        Attribute for which to get the value

    Returns
    -------
    :
        String version of the attribute's value
    """
    return f"{attribute}={getattr(instance, attribute)}"


def to_str(instance: Any, exposed_attributes: Iterable[str]) -> str:
    """
    Convert an instance to its string representation

    Parameters
    ----------
    instance
        Instance to convert

    exposed_attributes
        Attributes from Fortran that the instance exposes

    Returns
    -------
    :
        String representation of the instance
    """
    if not exposed_attributes:
        return repr(instance)

    attribute_values = [
        get_attribute_str_value(instance, v) for v in exposed_attributes
    ]

    return f"{repr(instance)[:-1]}, {', '.join(attribute_values)})"


def to_pretty(
    instance: Any,
    exposed_attributes: Iterable[str],
    p: Any,
    cycle: bool,
    indent: int = 4,
) -> None:
    """
    Pretty-print an instance

    Parameters
    ----------
    instance
        Instance to convert

    exposed_attributes
        Attributes from Fortran that the instance exposes

    p
        Pretty printing object

    cycle
        Whether the pretty printer has detected a cycle or not.

    indent
        Indent to apply to the pretty printing group
    """
    if not exposed_attributes:
        p.text(str(instance))
        return

    with p.group(indent, f"{repr(instance)[:-1]}", ")"):
        for att in exposed_attributes:
            p.text(",")
            p.breakable()

            p.text(get_attribute_str_value(instance, att))


def add_attribute_row(
    attribute_name: str, attribute_value: str, attribute_rows: list[str]
) -> list[str]:
    """
    Add a row for displaying an attribute's value to a list of rows

    Parameters
    ----------
    attribute_name
        Attribute's name

    attribute_value
        Attribute's value

    attribute_rows
        Existing attribute rows


    Returns
    -------
        Attribute rows, with the new row appended
    """
    attribute_rows.append(
        f"<tr><th>{attribute_name}</th><td style='text-align:left;'>{attribute_value}</td></tr>"  # noqa: E501
    )

    return attribute_rows


def to_html(instance: Any, exposed_attributes: Iterable[str]) -> str:
    """
    Convert an instance to its html representation

    Parameters
    ----------
    instance
        Instance to convert

    exposed_attributes
        Attributes from Fortran that the instance exposes

    Returns
    -------
    :
        HTML representation of the instance
    """
    if not exposed_attributes:
        return str(instance)

    instance_class_name = repr(instance).split("(")[0]

    attribute_rows: list[str] = []
    for att in exposed_attributes:
        att_val = getattr(instance, att)

        try:
            att_val = att_val._repr_html_()
        except AttributeError:
            att_val = str(att_val)

        attribute_rows = add_attribute_row(att, att_val, attribute_rows)

    attribute_rows_for_table = "\n          ".join(attribute_rows)

    css_style = """.fgen-wrap {
  /*font-family: monospace;*/
  width: 540px;
}

.fgen-header {
  padding: 6px 0 6px 3px;
  border-bottom: solid 1px #777;
  color: #555;;
}

.fgen-header > div {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.fgen-basefinalizable-cls,
.fgen-basefinalizable-instance-index {
  margin-left: 2px;
  margin-right: 10px;
}

.fgen-basefinalizable-cls {
  font-weight: bold;
  color: #000000;
}"""

    return "\n".join(
        [
            "<div>",
            "  <style>",
            f"{css_style}",
            "  </style>",
            "  <div class='fgen-wrap'>",
            "    <div class='fgen-header'>",
            "        <table><tbody>",
            f"          {attribute_rows_for_table}",
            "        </tbody></table>",
            "    </div>",
            "  </div>",
            "</div>",
        ]
    )


# %%
from enum import StrEnum
from typing import Protocol

import pint
from attrs import define


class InterpolatorLike(Protocol):
    """Interpolator-like"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """


@define
class InterpolatorPiecewiseConstantPreviousLeftInclusive:
    time: pint.UnitRegistry.Quantity  # array
    """Time points"""

    y_values: pint.UnitRegistry.Quantity  # array
    """Values"""

    allow_extrapolation: bool
    """Should extrapolation be allowed"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """
        # TODO: extrapolation checks
        res_idxs = (
            np.searchsorted(a=self.time, v=np.atleast_1d(time_target), side="right") - 1
        )
        # Fix up any overrun
        res_idxs[res_idxs == -1] = 0
        res = self.y_values[res_idxs]

        return res


@define
class InterpolatorPiecewiseConstantPreviousLeftExclusive:
    time: pint.UnitRegistry.Quantity  # array
    """Time points"""

    y_values: pint.UnitRegistry.Quantity  # array
    """Values"""

    allow_extrapolation: bool
    """Should extrapolation be allowed"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """
        res_idxs = (
            np.searchsorted(a=self.time, v=np.atleast_1d(time_target), side="left") - 1
        )
        # Fix up any overrun
        res_idxs[res_idxs == -1] = 0
        res = self.y_values[res_idxs]

        return res


@define
class InterpolatorPiecewiseConstantNextLeftInclusive:
    time: pint.UnitRegistry.Quantity  # array
    """Time points"""

    y_values: pint.UnitRegistry.Quantity  # array
    """Values"""

    allow_extrapolation: bool
    """Should extrapolation be allowed"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """
        res_idxs = np.searchsorted(
            a=self.time, v=np.atleast_1d(time_target), side="right"
        )
        # Fix up any overrun
        res_idxs[res_idxs == self.time.size] = self.time.size - 1
        res = self.y_values[res_idxs]

        return res


@define
class InterpolatorPiecewiseConstantNextLeftExclusive:
    time: pint.UnitRegistry.Quantity  # array
    """Time points"""

    y_values: pint.UnitRegistry.Quantity  # array
    """Values"""

    allow_extrapolation: bool
    """Should extrapolation be allowed"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """
        res_idxs = np.searchsorted(
            a=self.time, v=np.atleast_1d(time_target), side="left"
        )
        # Fix up any overrun
        res_idxs[res_idxs == self.time.size] = self.time.size - 1
        res = self.y_values[res_idxs]

        return res


@define
class InterpolatorLinear:
    time: pint.UnitRegistry.Quantity  # array
    """Time points"""

    y_values: pint.UnitRegistry.Quantity  # array
    """Values"""

    allow_extrapolation: bool
    """Should extrapolation be allowed"""

    # def interpolate(self, time_target: TimeAxis) -> pint.UnitRegistry.Quantity:  # array
    def interpolate(self, time_target) -> pint.UnitRegistry.Quantity:  # array
        """
        Interpolate

        Parameters
        ----------
        time_target
            Target time onto which to interpolate

        Returns
        -------
        :
            Interpolated values
        """
        coeffs = np.zeros((2, self.y_values.size - 1))
        coeffs[0, :] = (
            (self.y_values[1:] - self.y_values[:-1]) / (self.time[1:] - self.time[:-1])
        ).m
        coeffs[1, :] = self.y_values[:-1].m
        x = self.time.m

        ppoly = scipy.interpolate.PPoly(
            c=coeffs, x=x, extrapolate=self.allow_extrapolation
        )
        res = ppoly(time_target.m) * self.y_values.u

        return res


class InterpolationOption(StrEnum):
    """
    Interpolation options
    """

    NotSpecified = "not_specified"
    """No handling has been specified"""

    Linear = "linear"
    """Linear interpolation is assumed between points"""

    Quadratic = "quadratic"
    """Quadratic interpolation is assumed between points"""

    Cubic = "cubic"
    """Cubic interpolation is assumed between points"""

    PiecewiseConstantPreviousLeftInclusive = (
        "piecewise_constant_previous_left_inclusive"
    )
    """
    Between t(i) and t(i + 1), the value is equal to y(i)

    At t(i), the value is equal to y(i).
    """

    PiecewiseConstantPreviousLeftExclusive = (
        "piecewise_constant_previous_left_exclusive"
    )
    """
    Between t(i) and t(i + 1), the value is equal to y(i)

    At t(i), the value is equal to y(i - 1).
    """

    PiecewiseConstantNextLeftInclusive = "piecewise_constant_next_left_inclusive"
    """
    Between t(i) and t(i + 1), the value is equal to y(i + 1)

    At t(i), the value is equal to y(i + 1).
    """

    PiecewiseConstantNextLeftExclusive = "piecewise_constant_next_left_exclusive"
    """
    Between t(i) and t(i + 1), the value is equal to y(i + 1)

    At t(i), the value is equal to y(i).
    """


def create_interpolator(
    time: pint.UnitRegistry.Quantity,  # array
    y_values: pint.UnitRegistry.Quantity,  # array
    kind: InterpolationOption,
    allow_extrapolation: bool = False,
) -> InterpolatorLike:
    if kind == InterpolationOption.PiecewiseConstantPreviousLeftInclusive:
        return InterpolatorPiecewiseConstantPreviousLeftInclusive(
            time=time,
            y_values=y_values,
            allow_extrapolation=allow_extrapolation,
        )

    if kind == InterpolationOption.PiecewiseConstantPreviousLeftExclusive:
        return InterpolatorPiecewiseConstantPreviousLeftExclusive(
            time=time,
            y_values=y_values,
            allow_extrapolation=allow_extrapolation,
        )
    if kind == InterpolationOption.PiecewiseConstantNextLeftInclusive:
        return InterpolatorPiecewiseConstantNextLeftInclusive(
            time=time,
            y_values=y_values,
            allow_extrapolation=allow_extrapolation,
        )

    if kind == InterpolationOption.PiecewiseConstantNextLeftExclusive:
        return InterpolatorPiecewiseConstantNextLeftExclusive(
            time=time,
            y_values=y_values,
            allow_extrapolation=allow_extrapolation,
        )

    if kind == InterpolationOption.Linear:
        return InterpolatorLinear(
            time=time,
            y_values=y_values,
            allow_extrapolation=allow_extrapolation,
        )

    raise NotImplementedError(kind)


# %%
from typing import Any

import matplotlib.axes
import matplotlib.pyplot
import pint
from attrs import define, field


@define
class ValuesBounded:
    """
    Container for values that supports retrieving bounds too

    The bounds are defined by the values,
    with the `last_bound_value` attribute
    resolving any ambiguity about what value to use for the last bound.

    When handling time values, this lets us define the extent of the last time step.
    When handling other values, it provides a key piece of information needed
    to do interpolation/integration/differentiation unambiguosly.

    However, you must remember that this container is quite low-level.
    As a result, it does not provide all the information required to do operations,
    such as interpolation/integration/differentiation, unambiguosly.
    For example, it does not specify whether the bounds are
    open, closed, half-open, half-closed etc.
    That information has to come from other classes/information.
    For example, the kind of interpolation
    (which does specify whether the bounds are open, closed etc.
    see [`InterpolationOption`][] and related functionality).

    The design means that the bounds are tightly coupled to the values.
    This is deliberate, as it significantly simplifes our life.
    For example, one result of this construction is
    that the bounds are always contiguous.
    In other words, we can have a bounds concept and API
    without the headaches of having to handle arbitrary bounds,
    particularly those that have gaps.
    This is clearly a design trade-off,
    although one we think could be undone later if needed.

    One other consequence of this container's structure is
    that we can’t have bounds which start before the first value
    (in the case of time, this means the start of the first timestep
    is always equal to the first value).
    However, we can't think of a situation in which that is needed
    (and excluding this possibility makes life much simpler).
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

    def __str__(self) -> str:
        """
        Get string representation of self
        """
        return to_str(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """
        Get pretty representation of self

        Used by IPython notebooks and other tools
        """
        to_pretty(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
            p=p,
            cycle=cycle,
        )

    def _repr_html_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        return to_html(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    @property
    def all_values(self) -> pint.UnitRegistry.Quantity:  # array
        """
        Get the values, including `self.value_last_bound`

        Returns
        -------
        :
            Values, including `self.value_last_bound`
        """
        return np.hstack([self.values, self.value_last_bound])


@define
class TimeAxis:
    """
    Same idea as [`ValuesBounded`][], except the values must be strictly monotonically increasing.
    """

    values: pint.UnitRegistry.Quantity = field()  # array
    """
    Values
    """

    value_last_bound: pint.UnitRegistry.Quantity = field()  # scalar
    """
    Value to use for the last value to use in the bounds array

    Required to avoid ambiguity in this value
    """

    @values.validator
    def values_validator(
        self,
        # attribute: attr.Attribute[Any],
        attribute,
        value: pint.UnitRegistry.Quantity,  # array
    ) -> None:
        delta = value[1:] - value[:-1]
        if (delta <= 0).any():
            msg = f"values must be strictly monotonic, received: {value=}"
            raise ValueError(msg)

    @value_last_bound.validator
    def value_last_bound_validator(
        self,
        # attribute: attr.Attribute[Any],
        attribute,
        value: pint.UnitRegistry.Quantity,  # scalar
    ) -> None:
        if value <= self.values[-1]:
            msg = (
                "value must be greater than the last value in `self.values`. "
                f"Received: {value=}, {self.values[-1]=}"
            )
            raise ValueError(msg)

    def __str__(self) -> str:
        """
        Get string representation of self
        """
        return to_str(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """
        Get pretty representation of self

        Used by IPython notebooks and other tools
        """
        to_pretty(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
            p=p,
            cycle=cycle,
        )

    def _repr_html_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        return to_html(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    @property
    def bounds(self) -> pint.UnitRegistry.Quantity:  # array
        """
        Get the bounds of the time steps

        Returns
        -------
        :
            Bounds of the time steps
        """
        return np.hstack([self.values, self.value_last_bound])

    @property
    def bounds_2d(self) -> pint.UnitRegistry.Quantity:  # array[..., 2]
        """
        Get the bounds of the time steps in two-dimensions

        Can be useful for some operations

        Returns
        -------
        :
            Bounds of the time steps in two-dimensions
            (bounds is the second dimension i.e. has size 2).
        """
        starts = self.values
        ends = np.hstack([self.values[1:], self.value_last_bound])
        return np.vstack([starts, ends]).T


@define
class Timeseries:
    """
    Representation of a continuous timeseries

    At the moment, this only supports one-dimensional timeseries.
    """

    name: str
    """Name of the timeseries"""

    time: TimeAxis
    """Time axis of the timeseries"""

    values: ValuesBounded
    """Values that define the timeseries"""

    interpolation: InterpolationOption
    """Interpolation to apply to the timeseries"""

    # Change API to just carry this around always.
    # Make `from_values` a class method or something
    # so user doesn't have to make it.
    # Validation would be checking that the polynomial,
    # evaluated at `self.time` gives the expected values.
    # This could also be a separate method
    # (e.g. `self.validate_polynomial_values_consistency`).
    # Or, just don't carry the stuff you can infer around
    # and only use that for __str__, __repr__ etc.
    piecewise_polynomial: scipy.interpolate.PPoly | None = None
    """
    If supplied, the piecewise polynomial that represents this timeseries.

    If not supplied, we will create this as needed.
    """

    def __str__(self) -> str:
        """
        Get string representation of self
        """
        return to_str(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """
        Get pretty representation of self

        Used by IPython notebooks and other tools
        """
        to_pretty(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
            p=p,
            cycle=cycle,
        )

    def _repr_html_(self) -> str:
        """
        Get html representation of self

        Used by IPython notebooks and other tools
        """
        return to_html(
            self,
            tuple(a.name for a in self.__attrs_attrs__),
        )

    @property
    def values_at_bounds(self) -> pint.UnitRegistry.Quantity:  # array
        """
        Get the values at the bounds defined by `self.time`

        Returns
        -------
        :
            Values at the bounds defined by `self.time`
        """
        return self.interpolate(self.time).values.all_values

    @classmethod
    def from_integration_result(cls, integration_result, name: str):
        # def from_integration_result(cls, integration_result: IntegrationResult, name: str) -> Timeseries:
        time = TimeAxis(
            values=integration_result.time_bounds[:-1],
            value_last_bound=integration_result.time_bounds[-1],
        )
        values = ValuesBounded(
            values=integration_result.integral_at_bounds[:-1],
            value_last_bound=integration_result.integral_at_bounds[-1],
        )

        return cls(
            name=name,
            time=time,
            values=values,
            interpolation=integration_result.interpolation,
            piecewise_polynomial=integration_result.piecewise_polynomial,
        )

    # def interpolate(self, time_axis_new: TimeAxis, allow_extrapolation: bool = False) -> Timeseries:
    def interpolate(self, time_axis_new: TimeAxis, allow_extrapolation: bool = False):
        if self.piecewise_polynomial is not None:
            values_interp = (
                self.piecewise_polynomial(time_axis_new.values.m)
                * time_axis_new.values.u
            )
            value_last_bound_interp = (
                self.piecewise_polynomial(time_axis_new.value_last_bound.m)
                * time_axis_new.value_last_bound.u
            )

        else:
            interpolator = self.get_interpolator(
                allow_extrapolation=allow_extrapolation
            )

            values_interp = interpolator.interpolate(time_axis_new.values)
            value_last_bound_interp = interpolator.interpolate(
                time_axis_new.value_last_bound
            )

        res = Timeseries(
            name=self.name,
            time=time_axis_new,
            values=ValuesBounded(
                values=values_interp,
                value_last_bound=value_last_bound_interp,
            ),
            interpolation=self.interpolation,
            piecewise_polynomial=self.piecewise_polynomial,
        )

        return res

    # def interpolate(self, time_axis_new: TimeAxis, allow_extrapolation: bool = False) -> Timeseries:
    def integrate(
        self,
        integration_constant: pint.UnitRegistry.Quantity,  # scalar
        name_res: str | None = None,
    ):
        # ) -> Timeseries:
        if name_res is None:
            name_res = f"{self.name}_integral"

        integrate_res = integrate(
            time_bounds=self.time.bounds,
            y_at_bounds=self.values_at_bounds,
            interpolation=self.interpolation,
            integration_constant=integration_constant,
        )

        return type(self).from_integration_result(integrate_res, name=name_res)

    def get_interpolator(self, allow_extrapolation: bool = False) -> InterpolatorLike:
        time_interp = self.time.bounds
        y_interp = self.values.all_values

        interpolator = create_interpolator(
            time=time_interp,
            y_values=y_interp,
            kind=self.interpolation,
            allow_extrapolation=allow_extrapolation,
        )

        return interpolator

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show_discrete: bool = False,
        res_increase: int = 1000,
        plot_kwargs: dict[str, Any] | None = None,
        discrete_kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if plot_kwargs is None:
            plot_kwargs = {}

        if discrete_kwargs is None:
            discrete_kwargs = {}

        # Interpolate.
        # Then plot interpolated using linear joins
        # (as far as I can tell, this is the only general way to do this,
        # although it is slower than using e.g. step for piecewise constant stuff).)
        show_time_points = (
            np.unique(
                np.concatenate(
                    [
                        np.linspace(
                            self.time.values[0].m,
                            self.time.value_last_bound.m,
                            res_increase,
                        ),
                        self.time.bounds.m,
                    ]
                )
            )
            * self.time.values.u
        )

        show_time_points = np.sort(show_time_points)
        interpolated_times = TimeAxis(
            values=show_time_points[:-1],
            value_last_bound=self.time.value_last_bound,
        )

        interpolated = self.interpolate(interpolated_times)
        ax.plot(
            interpolated.time.bounds.m,
            interpolated.values_at_bounds.m,
            **plot_kwargs,
        )

        if show_discrete:
            ax.scatter(
                self.time.bounds.m,
                self.values_at_bounds.m,
                **discrete_kwargs,
            )
            # add difference between values and value_last_bound?

        return ax


# %%
TimeAxis(Q([1, 2, 3], "yr"), Q(4, "yr"))
# TimeAxis(Q([1, 2, 3], "yr"), Q(2, "yr"))
# TimeAxis(Q([1, 2, 3], "yr"), Q(3, "yr"))
# TimeAxis(Q([1, 2, 1], "yr"), Q(4, "yr"))
# TimeAxis(Q([1, 2, 2], "yr"), Q(4, "yr"))

# %%
print(Timeseries.__doc__)

# %% [markdown]
# The API provided by `Timeseries` offers a solution to the problems above.
# The tradeoff is that you have to think about things which you wouldn't normally consider.

# %% [markdown]
# In the case of our COVID emissions example, we would capture the timeseries we want as shown below.

# %%
covid_emissions = Timeseries(
    name="co2_emissions",
    time=TimeAxis(
        values=Q(np.array([2018, 2019, 2020, 2021, 2022, 2023]), "yr"),
        value_last_bound=Q(2024, "yr"),
    ),
    values=ValuesBounded(
        values=Q(np.array([10.2, 10.3, 9.5, 10.1, 10.3, 10.5]), "GtC / yr"),
        value_last_bound=Q(10.6, "GtC / yr"),
    ),
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftInclusive,
)
covid_emissions

# %%
# # Use something like this for testing interpolation in various forms
# time_axis_new_example = TimeAxis(
#     values=Q(np.array([2018, 2018.5, 2019.0, 2019.5, 2020.0]), "yr"),
#     value_last_bound=Q(2020.5, "yr")
# )
# covid_emissions.interpolate(time_axis_new=time_axis_new_example)

# %% [markdown]
# We can visualise the implications of different interpolation options like the below.

# %%
covid_emissions

# %%
fig, ax = plt.subplots(figsize=(12, 8))

for interp_option, marker in (
    (InterpolationOption.PiecewiseConstantPreviousLeftInclusive, "o"),
    (InterpolationOption.PiecewiseConstantPreviousLeftExclusive, "o"),
    (InterpolationOption.PiecewiseConstantNextLeftInclusive, "x"),
    (InterpolationOption.PiecewiseConstantNextLeftExclusive, "x"),
    (InterpolationOption.Linear, "v"),
):
    evolve(covid_emissions, interpolation=interp_option).plot(
        ax=ax,
        show_discrete=True,
        plot_kwargs=dict(alpha=0.4, label=interp_option),
        discrete_kwargs=dict(alpha=0.4, label=interp_option, marker=marker, s=130),
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.grid()

# %% [markdown]
# These different interpolation options have a much clearer impact once we interpolate.
#
# For this, we also pick slightly different emissions.

# %%
integration_demo_emissions = Timeseries(
    name="co2_emissions",
    time=TimeAxis(
        values=Q(np.array([1850, 1900, 2000]), "yr"), value_last_bound=Q(2100, "yr")
    ),
    values=ValuesBounded(
        values=Q(np.array([0, 10.0, 0.0]), "GtC / yr"),
        value_last_bound=Q(2.5, "GtC / yr"),
    ),
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftInclusive,
)

# %%
from attrs import define


@define
class IntegrationResult:
    """Result of performing an integration"""

    time_bounds: pint.UnitRegistry.Quantity  # array
    """Time bounds on which the integration was performed"""

    integral_at_bounds: pint.UnitRegistry.Quantity  # array
    """Integral value at the boudns defined by time_bounds"""

    interpolation: InterpolationOption
    """Interpolation that applies to the integration result"""

    piecewise_polynomial: scipy.interpolate.PPoly | None = None
    """
    The piecewise polynomial that represents the integration result
    """


def integrate_piecewise_constant_previous_left_inclusive(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    time_step = time_bounds[1:] - time_bounds[:-1]
    # Previous left-inclusive interpolation so can ignore y_at_bounds[-1]
    window_integrals = y_at_bounds[:-1] * time_step

    integral_at_bounds = np.hstack(
        [integration_constant, np.cumsum(window_integrals) + integration_constant]
    )

    res = IntegrationResult(
        time_bounds=time_bounds,
        integral_at_bounds=integral_at_bounds,
        interpolation=InterpolationOption.Linear,
    )

    return res


def integrate_piecewise_constant_previous_left_exclusive(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    time_step = time_bounds[1:] - time_bounds[:-1]
    # Previous left-exclusive interpolation so can ignore y_at_bounds[0]
    window_integrals = y_at_bounds[1:] * time_step

    integral_at_bounds = np.hstack(
        [integration_constant, np.cumsum(window_integrals) + integration_constant]
    )

    res = IntegrationResult(
        time_bounds=time_bounds,
        integral_at_bounds=integral_at_bounds,
        interpolation=InterpolationOption.Linear,
    )

    return res


def integrate_piecewise_constant_next_left_inclusive(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    time_step = time_bounds[1:] - time_bounds[:-1]
    # Next left-inclusive interpolation so can ignore y_at_bounds[-1]
    window_integrals = y_at_bounds[:-1] * time_step

    integral_at_bounds = np.hstack(
        [integration_constant, np.cumsum(window_integrals) + integration_constant]
    )

    res = IntegrationResult(
        time_bounds=time_bounds,
        integral_at_bounds=integral_at_bounds,
        interpolation=InterpolationOption.Linear,
    )

    return res


def integrate_piecewise_constant_next_left_exclusive(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    time_step = time_bounds[1:] - time_bounds[:-1]
    # Next left-exclusive interpolation so can ignore y_at_bounds[0]
    window_integrals = y_at_bounds[1:] * time_step

    integral_at_bounds = np.hstack(
        [integration_constant, np.cumsum(window_integrals) + integration_constant]
    )

    res = IntegrationResult(
        time_bounds=time_bounds,
        integral_at_bounds=integral_at_bounds,
        interpolation=InterpolationOption.Linear,
    )

    return res


def integrate_linear(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    # TODO: split out `create_linear` or something somewhere
    coeffs = np.zeros((2, y_at_bounds.size - 1))
    coeffs[0, :] = (
        (y_at_bounds[1:] - y_at_bounds[:-1]) / (time_bounds[1:] - time_bounds[:-1])
    ).m
    coeffs[1, :] = y_at_bounds[:-1].m
    x = time_bounds.m

    ppoly = scipy.interpolate.PPoly(c=coeffs, x=x, extrapolate=False)
    tmp = ppoly.antiderivative()
    c_new = tmp.c
    c_new[2, :] += integration_constant.m
    indefinite_integral = scipy.interpolate.PPoly(
        c=c_new, x=tmp.x, extrapolate=tmp.extrapolate
    )

    integral_at_bounds = (
        indefinite_integral(time_bounds.m) * y_at_bounds.u * time_bounds.u
    )

    res = IntegrationResult(
        time_bounds=time_bounds,
        integral_at_bounds=integral_at_bounds,
        interpolation=InterpolationOption.Quadratic,
        piecewise_polynomial=indefinite_integral,
    )

    return res


def integrate(
    time_bounds: pint.UnitRegistry.Quantity,  # array
    y_at_bounds: pint.UnitRegistry.Quantity,  # array
    interpolation: InterpolationOption,
    integration_constant: pint.UnitRegistry.Quantity,  # scalar
) -> IntegrationResult:
    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftInclusive:
        return integrate_piecewise_constant_previous_left_inclusive(
            time_bounds=time_bounds,
            y_at_bounds=y_at_bounds,
            integration_constant=integration_constant,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftExclusive:
        return integrate_piecewise_constant_previous_left_exclusive(
            time_bounds=time_bounds,
            y_at_bounds=y_at_bounds,
            integration_constant=integration_constant,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftInclusive:
        return integrate_piecewise_constant_next_left_inclusive(
            time_bounds=time_bounds,
            y_at_bounds=y_at_bounds,
            integration_constant=integration_constant,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftExclusive:
        return integrate_piecewise_constant_next_left_exclusive(
            time_bounds=time_bounds,
            y_at_bounds=y_at_bounds,
            integration_constant=integration_constant,
        )

    if interpolation == InterpolationOption.Linear:
        return integrate_linear(
            time_bounds=time_bounds,
            y_at_bounds=y_at_bounds,
            integration_constant=integration_constant,
        )

    raise NotImplementedError(interpolation)


# %%
fig, axes = plt.subplots(figsize=(12, 8), nrows=2)

for interp_option, marker in (
    (InterpolationOption.PiecewiseConstantPreviousLeftInclusive, "o"),
    (InterpolationOption.PiecewiseConstantPreviousLeftExclusive, "o"),
    (InterpolationOption.PiecewiseConstantNextLeftInclusive, "x"),
    (InterpolationOption.PiecewiseConstantNextLeftExclusive, "x"),
    (InterpolationOption.Linear, "v"),
):
    evolve(integration_demo_emissions, interpolation=interp_option).plot(
        ax=axes[0],
        show_discrete=True,
        plot_kwargs=dict(alpha=0.4, label=interp_option),
        discrete_kwargs=dict(alpha=0.4, label=interp_option, marker=marker, s=130),
    )

    evolve(integration_demo_emissions, interpolation=interp_option).integrate(
        integration_constant=Q(500, "GtC")
        # integration_constant=Q(0, "GtC")
    ).plot(
        ax=axes[1],
        show_discrete=True,
        plot_kwargs=dict(alpha=0.4, label=interp_option),
        discrete_kwargs=dict(alpha=0.4, label=interp_option, marker=marker, s=130),
    )

for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

# %%
