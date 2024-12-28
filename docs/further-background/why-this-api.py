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
from typing import Any

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

    PiecewiseConstantPrevious = "piecewise_constant_previous"
    """The value is equal to the last defined point"""

    PiecewiseConstantNext = "piecewise_constant_next"
    """The value is equal to the next defined point"""


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
    interpolation=InterpolationOption.PiecewiseConstantPrevious,
)
covid_emissions

# %%
import matplotlib.axes
import matplotlib.pyplot


# %%
def plot(
    self,
    ax: matplotlib.axes.Axes | None = None,
    show_discrete: bool = False,
) -> matplotlib.axes.Axes:
    if ax is None:
        _, ax = plt.subplots()

    # interpolate
    # plot interpolated
    # (only general way to do this, although it is slower)

    if show_discrete:
        ax.scatter(
            self.time.bounds.m,
            self.values.all_values.m,
        )
        # add difference between values and value_last_bound?

    return ax


# %%
fig, ax = plt.subplots()
# plot(covid_emissions, ax=ax)
plot(covid_emissions, ax=ax, show_discrete=True)
# covid_emissions.plot(ax=ax)
ax.grid()

# %%
emissions_covid = Q(np.array([10.2, 10.3, 9.5, 10.1, 10.3, 10.5]), "GtC / yr")
years_covid = Q(np.array([2018, 2019, 2020, 2021, 2022, 2023]), "yr")