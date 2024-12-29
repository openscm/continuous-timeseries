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
from enum import StrEnum


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

    PiecewiseConstantPreviousLeftClosed = "piecewise_constant_previous_left_closed"
    """
    Between t(i) and t(i + 1), the value is equal to y(i)

    At t(i), the value is equal to y(i).
    """

    PiecewiseConstantPreviousLeftOpen = "piecewise_constant_previous_left_open"
    """
    Between t(i) and t(i + 1), the value is equal to y(i)

    At t(i), the value is equal to y(i - 1).
    """

    PiecewiseConstantNextLeftClosed = "piecewise_constant_next_left_closed"
    """
    Between t(i) and t(i + 1), the value is equal to y(i + 1)

    At t(i), the value is equal to y(i + 1).
    """

    PiecewiseConstantNextLeftOpen = "piecewise_constant_next_left_open"
    """
    Between t(i) and t(i + 1), the value is equal to y(i + 1)

    At t(i), the value is equal to y(i).
    """


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

    # def __str__(self) -> str:
    #     """
    #     Get string representation of self
    #     """
    #     return to_str(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    # def _repr_pretty_(self, p: Any, cycle: bool) -> None:
    #     """
    #     Get pretty representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     to_pretty(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #         p=p,
    #         cycle=cycle,
    #     )

    # def _repr_html_(self) -> str:
    #     """
    #     Get html representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     return to_html(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

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

    @classmethod
    def from_all_values(
        cls,
        all_values: pint.UnitRegistry.Quantity,  # array
    ):
        return cls(
            values=all_values[:-1],
            value_last_bound=all_values[-1],
        )


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

    # def __str__(self) -> str:
    #     """
    #     Get string representation of self
    #     """
    #     return to_str(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    # def _repr_pretty_(self, p: Any, cycle: bool) -> None:
    #     """
    #     Get pretty representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     to_pretty(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #         p=p,
    #         cycle=cycle,
    #     )

    # def _repr_html_(self) -> str:
    #     """
    #     Get html representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     return to_html(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

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

    @classmethod
    def from_bounds(
        cls,
        bounds: pint.UnitRegistry.Quantity,  # array
    ):
        return cls(
            values=bounds[:-1],
            value_last_bound=bounds[-1],
        )


@define
class TimeseriesDiscrete:
    """
    Representation of a discrete timeseries

    At the moment, this only supports one-dimensional timeseries.
    """

    name: str
    """Name of the timeseries"""

    time: TimeAxis
    """Time axis of the timeseries"""

    values: ValuesBounded
    """Values that define the timeseries"""

    # def __str__(self) -> str:
    #     """
    #     Get string representation of self
    #     """
    #     return to_str(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    # def _repr_pretty_(self, p: Any, cycle: bool) -> None:
    #     """
    #     Get pretty representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     to_pretty(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #         p=p,
    #         cycle=cycle,
    #     )

    # def _repr_html_(self) -> str:
    #     """
    #     Get html representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     return to_html(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    def to_continuous_timeseries(
        self,
        interpolation: InterpolationOption,
    ):
        # ) -> TimeseriesContinuous:
        return discrete_to_continuous(
            discrete=self,
            interpolation=interpolation,
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        different_value_last_bound: bool = False,
        value_last_bound_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if value_last_bound_kwargs is None:
            value_last_bound_kwargs = {}

        if different_value_last_bound:
            ax.scatter(
                self.time.values.m,
                self.values.values.m,
                **kwargs,
            )

            ax.scatter(
                self.time.value_last_bound.m,
                self.values.value_last_bound.m,
                **value_last_bound_kwargs,
            )

        else:
            ax.scatter(
                self.time.bounds.m,
                self.values.all_values.m,
                **kwargs,
            )

        return ax


# %%
@define
class TimeseriesContinuous:
    """
    Representation of a continous timeseries

    At the moment, this only supports one-dimensional timeseries.
    """

    name: str
    """Name of the timeseries"""

    time_units: pint.UnitRegistry.Unit
    """The units of the time axis"""

    values_units: pint.UnitRegistry.Unit
    """The units of the values"""

    piecewise_polynomial: scipy.interpolate.PPoly
    """
    The piecewise polynomial that represents this timeseries.
    """

    # def __str__(self) -> str:
    #     """
    #     Get string representation of self
    #     """
    #     return to_str(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    # def _repr_pretty_(self, p: Any, cycle: bool) -> None:
    #     """
    #     Get pretty representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     to_pretty(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #         p=p,
    #         cycle=cycle,
    #     )

    # def _repr_html_(self) -> str:
    #     """
    #     Get html representation of self

    #     Used by IPython notebooks and other tools
    #     """
    #     return to_html(
    #         self,
    #         tuple(a.name for a in self.__attrs_attrs__),
    #     )

    def to_discrete_timeseries(
        self,
        time_axis: TimeAxis,
    ):
        # ) -> TimeseriesDiscrete:
        ...

    def interpolate(
        self,
        times: TimeAxis | pint.UnitRegistry.Quantity,  # arrray
        allow_extrapolation: bool = False,
    ):
        if isinstance(times, TimeAxis):
            times = times.bounds

        times_m = times.to(self.time_units).m
        values_m = self.piecewise_polynomial(times_m, extrapolate=allow_extrapolation)
        if np.isnan(values_m).any():
            msg = f"Extrapolation when not allowed? ({allow_extrapolation=})"
            raise ValueError(msg)

        res = values_m * self.values_units

        return res

    def integrate(
        self,
        integration_constant: pint.UnitRegistry.Quantity,  # scalar
        name_res: str | None = None,
    ):
        if name_res is None:
            name_res = f"{self.name}_integral"

        values_units_integral = self.values_units * self.time_units

        indefinite_integral = self.piecewise_polynomial.antiderivative()

        c_new = indefinite_integral.c
        c_new[-1, :] += integration_constant.to(values_units_integral).m

        piecewise_polynomial_integral = scipy.interpolate.PPoly(
            c=c_new,
            x=indefinite_integral.x,
            extrapolate=False,
        )

        return TimeseriesContinuous(
            name=name_res,
            time_units=self.time_units,
            values_units=values_units_integral,
            piecewise_polynomial=piecewise_polynomial_integral,
        )

    def differentiate(
        self,
        name_res: str | None = None,
    ):
        if name_res is None:
            name_res = f"{self.name}_derivative"

        piecewise_polynomial_derivative = self.piecewise_polynomial.derivative()

        values_units_derivative = self.values_units / self.time_units

        return TimeseriesContinuous(
            name=name_res,
            time_units=self.time_units,
            values_units=values_units_derivative,
            piecewise_polynomial=piecewise_polynomial_derivative,
        )

    def plot(
        self,
        times: TimeAxis | pint.UnitRegistry.Quantity,  # array
        ax: matplotlib.axes.Axes | None = None,
        res_increase: int = 500,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if isinstance(times, TimeAxis):
            times = times.bounds

        # Interpolate.
        # Then plot interpolated using linear joins
        # (as far as I can tell, this is the only general way to do this,
        # although it is slower than using e.g. step for piecewise constant stuff).)
        show_time_points = (
            np.sort(
                np.unique(
                    np.concatenate(
                        [
                            np.linspace(
                                times[0].m,
                                times[-1].m,
                                times.size * res_increase,
                            ),
                            times.m,
                        ]
                    )
                )
            )
            * times.u
        )
        interpolated_values = self.interpolate(show_time_points)

        ax.plot(
            show_time_points.m,
            interpolated_values.m,
            **kwargs,
        )

        return ax


# %%
TimeAxis(Q([1, 2, 3], "yr"), Q(4, "yr"))
# Test cases for failing validation
# TimeAxis(Q([1, 2, 3], "yr"), Q(2, "yr"))
# TimeAxis(Q([1, 2, 3], "yr"), Q(3, "yr"))
# TimeAxis(Q([1, 2, 1], "yr"), Q(4, "yr"))
# TimeAxis(Q([1, 2, 2], "yr"), Q(4, "yr"))

# %% [markdown]
# The API provided by `TimeseriesDiscrete`, `TimeseriesContinous` and `Timeseries` offers a solution to the problems above.
# The tradeoff is that you have to think about things which you wouldn't normally consider.

# %% [markdown]
# In the case of our COVID emissions example, we would capture the timeseries we want as shown below.

# %%
covid_emissions = TimeseriesDiscrete(
    name="co2_emissions",
    time=TimeAxis(
        values=Q(np.array([2018, 2019, 2020, 2021, 2022, 2023]), "yr"),
        value_last_bound=Q(2024, "yr"),
    ),
    values=ValuesBounded(
        values=Q(np.array([10.2, 10.3, 9.5, 10.1, 10.3, 10.5]), "GtC / yr"),
        value_last_bound=Q(10.6, "GtC / yr"),
    ),
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
class PPolyPiecewiseConstantPreviousLeftClosed(scipy.interpolate.PPoly):
    """
    Previous piecewise constant class, where the interval is closed on the left

    In other words, the value at the left of each bound is taken from the window it belongs to.

    Provided to allow easy integration with the rest of the [`scipy.interpolate.PPoly`][] universe.
    """

    def __init__(self, c, x, value_last_bound, extrapolate=None, axis=0):
        if c.shape[0] != 1:
            msg = "Should only be used for piecewise constant polynomials"
            raise AssertionError(msg)

        super().__init__(c, x, extrapolate, axis)
        self._value_last_bound = value_last_bound

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        return scipy.interpolate.PPoly.construct_fast(c, x, extrapolate=None, axis=0)

    def _evaluate(self, x, nu, extrapolate, out):
        from scipy.interpolate import _ppoly

        _ppoly.evaluate(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x,
            x,
            nu,
            bool(extrapolate),
            out,
        )

        # Should test this carefully, not sure if we will hit the issue described here or not
        # https://stackoverflow.com/a/32191125
        x_on_border = np.isin(x, self.x)
        if x_on_border.any():
            idxs_self_x = np.searchsorted(a=self.x, v=x[x_on_border])

            n_window_values = self.c[0].size
            on_last_bound = idxs_self_x == n_window_values

            # Avoid indexing errors
            idxs_self_x_safe = np.copy(idxs_self_x)
            idxs_self_x_safe[on_last_bound] = n_window_values - 1

            overwrite_values = self.c[0][idxs_self_x_safe]
            overwrite_values[on_last_bound] = self._value_last_bound

            out[x_on_border] = overwrite_values[:, np.newaxis]


def discrete_to_continuous_piecewise_constant_previous_left_closed(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    coeffs = np.atleast_2d(discrete.values.all_values[:-1].m)
    value_last_bound = discrete.values.all_values[-1].m

    piecewise_polynomial = PPolyPiecewiseConstantPreviousLeftClosed(
        x=x,
        c=coeffs,
        value_last_bound=value_last_bound,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
class PPolyPiecewiseConstantPreviousLeftOpen(scipy.interpolate.PPoly):
    """
    Previous piecewise constant class, where the interval is open on the left

    In other words, the value at the left of each bound is taken from the previous window.

    Provided to allow easy integration with the rest of the [`scipy.interpolate.PPoly`][] universe.
    """

    def __init__(self, c, x, extrapolate=None, axis=0):
        if c.shape[0] != 1:
            msg = "Should only be used for piecewise constant polynomials"
            raise AssertionError(msg)

        super().__init__(c, x, extrapolate, axis)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        return scipy.interpolate.PPoly.construct_fast(c, x, extrapolate=None, axis=0)

    def _evaluate(self, x, nu, extrapolate, out):
        from scipy.interpolate import _ppoly

        _ppoly.evaluate(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x,
            x,
            nu,
            bool(extrapolate),
            out,
        )

        # Should test this carefully, not sure if we will hit the issue described here or not
        # https://stackoverflow.com/a/32191125
        x_on_border = np.isin(x, self.x)
        if x_on_border.any():
            idxs_self_x = np.searchsorted(a=self.x, v=x[x_on_border]) - 1
            # Avoid wrapping
            idxs_self_x[idxs_self_x == -1] = 0

            overwrite_values = self.c[0][idxs_self_x]

            out[x_on_border] = overwrite_values[:, np.newaxis]


def discrete_to_continuous_piecewise_constant_previous_left_open(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    coeffs = np.atleast_2d(discrete.values.all_values[:-1].m)

    piecewise_polynomial = PPolyPiecewiseConstantPreviousLeftOpen(
        x=x,
        c=coeffs,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
def discrete_to_continuous_piecewise_constant_next_left_closed(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    # Next left closed, so we can ignore the first value
    coeffs = np.atleast_2d(discrete.values.all_values[1:].m)

    # Can use the standard scipy API,
    # hence this is what everything will collapse back to
    # (e.g. if you differentiate a linear spline).
    piecewise_polynomial = scipy.interpolate.PPoly(
        x=x,
        c=coeffs,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
class PPolyPiecewiseConstantNextLeftOpen(scipy.interpolate.PPoly):
    """
    Next piecewise constant class, where the interval is open on the left

    In other words, the value at the left of each bound is taken from the previous window.

    Provided to allow easy integration with the rest of the [`scipy.interpolate.PPoly`][] universe.
    """

    def __init__(self, c, x, value_first_bound, extrapolate=None, axis=0):
        if c.shape[0] != 1:
            msg = "Should only be used for piecewise constant polynomials"
            raise AssertionError(msg)

        super().__init__(c, x, extrapolate, axis)
        self._value_first_bound = value_first_bound

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        return scipy.interpolate.PPoly.construct_fast(c, x, extrapolate=None, axis=0)

    def _evaluate(self, x, nu, extrapolate, out):
        from scipy.interpolate import _ppoly

        _ppoly.evaluate(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x,
            x,
            nu,
            bool(extrapolate),
            out,
        )

        # Should test this carefully, not sure if we will hit the issue described here or not
        # https://stackoverflow.com/a/32191125
        x_on_border = np.isin(x, self.x)
        if x_on_border.any():
            idxs_self_x = np.searchsorted(a=self.x, v=x[x_on_border])
            # "Next" interpolation
            idxs_self_x_use = idxs_self_x - 1
            overwrite_values = self.c[0][idxs_self_x_use]

            # Ensure correct value is used for first bound
            on_first_bound = idxs_self_x == 0
            overwrite_values[on_first_bound] = self._value_first_bound

            out[x_on_border] = overwrite_values[:, np.newaxis]


def discrete_to_continuous_piecewise_constant_next_left_open(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    all_values_m = discrete.values.all_values.m
    coeffs = np.atleast_2d(all_values_m[1:])
    # Next left exclusive so need the first value too
    value_first_bound = all_values_m[0]

    piecewise_polynomial = PPolyPiecewiseConstantNextLeftOpen(
        x=x,
        c=coeffs,
        value_first_bound=value_first_bound,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
# No closed vs. open differentiation available here yet i.e. assume contiguous
def discrete_to_continuous_linear(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m

    coeffs = np.zeros((2, discrete.values.all_values.size - 1))

    all_vals = discrete.values.all_values
    rises = all_vals[1:] - all_vals[:-1]
    time_bounds = discrete.time.bounds
    time_steps = time_bounds[1:] - time_bounds[:-1]
    coeffs[0, :] = (rises / time_steps).m

    coeffs[1, :] = discrete.values.values.m

    piecewise_polynomial = scipy.interpolate.PPoly(
        x=x,
        c=coeffs,
        extrapolate=False,  # Avoid extrapolation by default
    )

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
# No closed vs. open differentiation available here yet i.e. assume contiguous
def discrete_to_continuous_quadratic(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    y = discrete.values.all_values.m

    # Generally not a great idea to do this so blindly.
    # User can/should inject their own piecewise polynomial
    # if they want a more specific interpolation.
    tck = scipy.interpolate.splrep(x=x, y=y, k=2)
    piecewise_polynomial = scipy.interpolate.PPoly.from_spline(tck)

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


def discrete_to_continuous_cubic(
    discrete: TimeseriesDiscrete,
) -> TimeseriesContinuous:
    x = discrete.time.bounds.m
    y = discrete.values.all_values.m

    # Generally not a great idea to do this so blindly.
    # User can/should inject their own piecewise polynomial
    # if they want a more specific interpolation.
    tck = scipy.interpolate.splrep(x=x, y=y, k=3)
    piecewise_polynomial = scipy.interpolate.PPoly.from_spline(tck)

    res = TimeseriesContinuous(
        name=discrete.name,
        time_units=discrete.time.bounds.u,
        values_units=discrete.values.all_values.u,
        piecewise_polynomial=piecewise_polynomial,
    )

    return res


# %%
def discrete_to_continuous(
    discrete: TimeseriesDiscrete,
    interpolation: InterpolationOption,
) -> TimeseriesContinuous:
    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftClosed:
        return discrete_to_continuous_piecewise_constant_previous_left_closed(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantPreviousLeftOpen:
        return discrete_to_continuous_piecewise_constant_previous_left_open(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftClosed:
        return discrete_to_continuous_piecewise_constant_next_left_closed(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.PiecewiseConstantNextLeftOpen:
        return discrete_to_continuous_piecewise_constant_next_left_open(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.Linear:
        return discrete_to_continuous_linear(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.Quadratic:
        return discrete_to_continuous_quadratic(
            discrete=discrete,
        )

    if interpolation == InterpolationOption.Cubic:
        return discrete_to_continuous_cubic(
            discrete=discrete,
        )

    raise NotImplementedError(interpolation)


# %%
# # Helps catch unit stripped warnings
# import warnings
# warnings.filterwarnings("error")

# %%
fig, ax = plt.subplots(figsize=(12, 8))

covid_emissions.plot(
    ax=ax,
    label="Discrete points",
    different_value_last_bound=True,
    value_last_bound_kwargs=dict(label="Discrete point last bound"),
)
for interp_option, marker in (
    (InterpolationOption.PiecewiseConstantPreviousLeftClosed, "o"),
    (InterpolationOption.PiecewiseConstantPreviousLeftOpen, "o"),
    (InterpolationOption.PiecewiseConstantNextLeftClosed, "x"),
    (InterpolationOption.PiecewiseConstantNextLeftOpen, "x"),
    (InterpolationOption.Linear, "v"),
):
    continuous = covid_emissions.to_continuous_timeseries(interpolation=interp_option)
    continuous.plot(
        times=covid_emissions.time,
        ax=ax,
        alpha=0.4,
        label=interp_option,
        # res_increase=3000,
    )
    ax.scatter(
        covid_emissions.time.bounds.m,
        continuous.interpolate(covid_emissions.time).m,
        marker=marker,
        s=150,
        alpha=0.4,
        label=f"{interp_option} interpolated points",
        # continuous.interpolate(continuous.time),
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.grid()

# %% [markdown]
# These different interpolation options have a much clearer impact once we interpolate.
#
# For this, we also pick slightly different emissions.

# %%
integration_demo_emissions = TimeseriesDiscrete(
    name="co2_emissions",
    time=TimeAxis(
        values=Q(np.array([1850, 1900, 2000]), "yr"), value_last_bound=Q(2100, "yr")
    ),
    values=ValuesBounded(
        values=Q(np.array([0, 10.0, 0.0]), "GtC / yr"),
        value_last_bound=Q(2.5, "GtC / yr"),
    ),
)

# %%
fig, axes = plt.subplots(figsize=(12, 12), nrows=3)

integration_demo_emissions.plot(
    ax=axes[0],
    label="Discrete points",
    different_value_last_bound=True,
    value_last_bound_kwargs=dict(label="Discrete point last bound"),
)
for interp_option in (
    InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    InterpolationOption.PiecewiseConstantPreviousLeftOpen,
    InterpolationOption.PiecewiseConstantNextLeftClosed,
    InterpolationOption.PiecewiseConstantNextLeftOpen,
    InterpolationOption.Linear,
):
    continuous_rep = integration_demo_emissions.to_continuous_timeseries(
        interpolation=interp_option
    )

    continuous_rep.plot(
        times=integration_demo_emissions.time,
        ax=axes[0],
        alpha=0.4,
        label=f"{continuous_rep.name}__{interp_option}",
        res_increase=100,
    )

    integral = continuous_rep.integrate(integration_constant=Q(150, "GtC"))
    integral.plot(
        times=integration_demo_emissions.time,
        ax=axes[1],
        alpha=0.4,
        label=f"{integral.name}__{interp_option}",
        res_increase=100,
    )

    final_view = continuous_rep.differentiate()
    final_view.plot(
        times=integration_demo_emissions.time,
        ax=axes[2],
        alpha=0.4,
        label=f"{final_view.name}__{interp_option}",
        res_increase=100,
    )

for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

# %% [markdown]
# This is how you create a nice, sharp, stepwise forcing function.

# %%
stepwise_forcing = TimeseriesDiscrete(
    name="abrupt_forcing",
    time=TimeAxis(
        values=Q(np.array([1700, 1850, 2000]), "yr"), value_last_bound=Q(2100, "yr")
    ),
    values=ValuesBounded(
        values=Q(np.array([0.0, 4.0, 4.0]), "W / m^2"),
        value_last_bound=Q(4.0, "W / m^2"),
    ),
)
stepwise_forcing_continuous = stepwise_forcing.to_continuous_timeseries(
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed
)

fig, axes = plt.subplots(ncols=2)
stepwise_forcing_continuous.plot(stepwise_forcing.time, ax=axes[0])
stepwise_forcing_continuous.plot(Q(np.arange(1849, 1851, 0.01), "yr"), ax=axes[1])

for ax in axes:
    ax.grid()

stepwise_forcing_continuous.interpolate(
    Q([1849.5, 1849.9999, 1850.0, 1850.00001], "yr")
)

# %% [markdown]
# This is obviously quite different to interpolating this linearly.

# %%
oops_stepwise_forcing_continuous = stepwise_forcing.to_continuous_timeseries(
    interpolation=InterpolationOption.Linear
)

fig, axes = plt.subplots(ncols=2)
oops_stepwise_forcing_continuous.plot(stepwise_forcing.time, ax=axes[0])
oops_stepwise_forcing_continuous.plot(Q(np.arange(1849, 1851, 0.01), "yr"), ax=axes[1])

for ax in axes:
    ax.grid()

fig.tight_layout()

oops_stepwise_forcing_continuous.interpolate(
    Q([1849.5, 1849.9999, 1850.0, 1850.00001], "yr")
)

# %% [markdown]
# It's not solved properly with an initially smaller timestep either
# (you still get a leading or trailing edge that isn't sharp).

# %%
stepwise_forcing_annual_start = TimeseriesDiscrete(
    name="abrupt_forcing",
    time=TimeAxis(
        values=Q(np.array([1848, 1849, 1850, 1851]), "yr"),
        value_last_bound=Q(1852, "yr"),
    ),
    values=ValuesBounded(
        values=Q(np.array([0.0, 0.0, 4.0, 4.0]), "W / m^2"),
        value_last_bound=Q(4.0, "W / m^2"),
    ),
)
stepwise_forcing_annual_start_continuous = (
    stepwise_forcing_annual_start.to_continuous_timeseries(
        interpolation=InterpolationOption.Linear
    )
)

fig, ax = plt.subplots()
stepwise_forcing_annual_start_continuous.plot(stepwise_forcing_annual_start.time, ax=ax)

ax.grid()

stepwise_forcing_annual_start_continuous.interpolate(
    Q([1849.5, 1850.0, 1850.00001], "yr")
)


# %%
@define
class Timeseries:
    """Timeseries representation"""

    time: TimeAxis
    """Time axis of the timeseries"""

    continuous: TimeseriesContinuous
    """Continuous version of the timeseries"""

    # TODO: str, repr, html

    @property
    def name(self) -> str:
        """
        Name of the timeseries
        """
        return self.continuous.name

    @property
    def discrete(self) -> TimeseriesDiscrete:
        """
        Discrete view of the timeseries
        """
        values = ValuesBounded.from_all_values(self.continuous.interpolate(self.time))

        return TimeseriesDiscrete(
            name=self.name,
            time=self.time,
            values=values,
        )

    @classmethod
    def from_arrays(
        cls,
        all_values: pint.UnitRegistry.Quantity,  # array
        time_bounds: pint.UnitRegistry.Quantity,  # array
        interpolation: InterpolationOption,
        name: str,
    ):
        values = ValuesBounded.from_all_values(all_values)
        time = TimeAxis.from_bounds(time_bounds)

        discrete = TimeseriesDiscrete(
            name=name,
            time=time,
            values=values,
        )
        continuous = discrete_to_continuous(
            discrete=discrete,
            interpolation=interpolation,
        )

        return cls(
            time=time,
            continuous=continuous,
        )

    def differentiate(
        self,
        name_res: str | None = None,
    ):
        derivative = self.continuous.differentiate(
            name_res=name_res,
        )

        return type(self)(
            time=self.time,
            continuous=derivative,
        )

    def integrate(
        self,
        integration_constant: pint.UnitRegistry.Quantity,  # scalar
        name_res: str | None = None,
    ):
        integral = self.continuous.integrate(
            integration_constant=integration_constant,
            name_res=name_res,
        )

        return type(self)(
            time=self.time,
            continuous=integral,
        )

    def update_time(self, time: TimeAxis):
        # Should check here that times are compatible with extrapolation choices
        return type(self)(
            time=time,
            continuous=self.continuous,
        )

    def update_interpolation(self, interpolation: InterpolationOption):
        continuous = discrete_to_continuous(
            discrete=self.discrete,
            interpolation=interpolation,
        )

        return type(self)(
            time=self.time,
            continuous=continuous,
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show_continuous: bool = True,
        continuous_kwargs: dict[str, Any] | None = None,
        show_discrete: bool = False,
        discrete_kwargs: dict[str, Any] | None = None,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if continuous_kwargs is None:
            continuous_kwargs = {}

        if discrete_kwargs is None:
            discrete_kwargs = {}

        if show_continuous:
            if "label" not in continuous_kwargs:
                continuous_kwargs["label"] = self.name

            self.continuous.plot(
                ax=ax,
                times=self.time,
                **continuous_kwargs,
            )

        if show_discrete:
            if "label" not in discrete_kwargs:
                discrete_kwargs["label"] = f"{self.name} discrete points"

            self.discrete.plot(
                ax=ax,
                **discrete_kwargs,
            )

        if set_xlabel:
            ax.set_xlabel(self.continuous.time_units)

        if set_ylabel:
            ax.set_ylabel(self.continuous.values_units)

        return ax


# %%
base = Timeseries.from_arrays(
    all_values=Q([1, 2, 10, 20], "m"),
    time_bounds=Q([1750, 1850, 1900, 2000], "yr"),
    interpolation=InterpolationOption.Linear,
    name="base",
)

# %%
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))


base = Timeseries.from_arrays(
    all_values=Q([1, 2, 10, 20], "m"),
    time_bounds=Q([1750, 1850, 1900, 2000], "yr"),
    interpolation=InterpolationOption.Linear,
    name="base",
)
base.plot(
    ax=axes[0],
    show_continuous=False,
    show_discrete=True,
    set_xlabel=True,
    set_ylabel=True,
)
for interp_option in (
    InterpolationOption.Linear,
    InterpolationOption.Quadratic,
    InterpolationOption.Cubic,
    InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    InterpolationOption.PiecewiseConstantNextLeftClosed,
    InterpolationOption.PiecewiseConstantPreviousLeftOpen,
    InterpolationOption.PiecewiseConstantNextLeftOpen,
):
    ts = base.update_interpolation(interp_option)
    ts.plot(
        ax=axes[0],
        continuous_kwargs=dict(
            label=interp_option,
            res_increase=1000,
            # res_increase=30,
            alpha=0.4,
        ),
    )
    ts.integrate(Q(3, "m yr")).plot(
        ax=axes[1],
        continuous_kwargs=dict(
            label=f"integrated {interp_option}",
            res_increase=1000,
            alpha=0.4,
        ),
        set_xlabel=True,
        set_ylabel=True,
    )
    ts.differentiate().plot(
        ax=axes[2],
        continuous_kwargs=dict(
            label=f"differentiated {interp_option}",
            res_increase=1000,
            alpha=0.4,
        ),
        set_xlabel=True,
        set_ylabel=True,
    )

for ax in axes:
    ax.grid()
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

fig.tight_layout()

# %%
current_time = Q(2025.0, "yr")
current_emms = Q(10.0, "GtC / yr")
budget = Q(23.2, "GtC")

# %%
# We're solving for cumulative emissions, y
# as a function of time, x.
# Quadratic cumulative emissions means linear
# emissions.
#     y = -alpha * (x - x_nz)**2 + h
#
# where x_nz is the net-zero time
# and alpha is a constant
# (units [cumulative emissions] / [time] ** 2).
#
# At net-zero, emissions should be zero
# i.e. the gradient of cumulative emissions should be zero.
# This is satisfied by construction:
#     dy/dx(x_nz) = -2 * alpha * (x_nz - x_nz) = 0
#
# At net-zero, cumulative emissions should be equal to the budget:
#     y(x_nz) = -alpha * (x_nz - x_nz)**2 + h = budget
#     therefore h = budget    (1)
#
# At the starting point (normally today), x_0,
# cumulative emissions are zero:
#     y(x_0) = -alpha * (x_0 - x_nz)**2 + h = 0
#     therefore alpha * (x_0 - x_nz)**2 = h    (2)
#
# At the starting point,
# the gradient of cumulative emissions is also known:
#     dy/dx(x_0) = -2 * alpha * (x_0 - x_nz) = e_0
#     therefore (x_0 - x_nz) = -e_0 / (2 * alpha)    (3)
#
# Substituting (3) into (2):
#     alpha * (-e_0 / (2 * alpha))**2 = h
#     therefore
#     e_0**2 / (4 * alpha) = h
#     alpha = e_0**2 / (4 * h)
#
# which can then be used, in combination with (3), to solve for x_nz
#     x_nz = x_0 + e_0 / (2 * alpha)    (3)

h = budget
alpha = current_emms**2 / (4 * budget)
x_nz = current_time + current_emms / (2 * alpha)
x_nz

# %%
x = current_time
-alpha * (current_time - x_nz) ** 2 + h
-alpha * (x_nz - x_nz) ** 2 + h
-alpha * (Q(2050, "yr") - x_nz) ** 2 + h

# %%
# Convert into coefficients for our polynomial
#    y = -alpha * (x - x_nz)**2 + h
#    y = -alpha * x**2 + 2 * alpha * x_nz * x - alpha * x_nz ** 2 + h
#    y = -alpha * (x - x_0 + x_0)**2 + 2 * alpha * x_nz * (x - x_0 + x_0) - alpha * x_nz ** 2 + h
#    y = -alpha * (x - x_0) ** 2 - 2 * alpha * (x - x_0) * x_0 - alpha * x_0**2 + 2 * alpha * x_nz * (x - x_0) + 2 * alpha * x_nz * x_0 - alpha * x_nz ** 2 + h
#    y = -alpha * (x - x_0) ** 2 - 2 * alpha * x_0 * (x - x_0) + 2 * alpha * x_nz * (x - x_0) - alpha * x_0**2 + 2 * alpha * x_nz * x_0 - alpha * x_nz ** 2 + h
#    y = -alpha * (x - x_0) ** 2 + 2 * alpha * (x_nz - x_0) * (x - x_0) + h - alpha * x_0**2 + 2 * alpha * x_nz * x_0 - alpha * x_nz ** 2
#
# Therefore, coefficients in y = a * (x - x_0)**2 + b * (x - x_0) + c are:
#    a = -alpha
#    b = 2 * alpha * (x_nz - x_0)
#    c = h - alpha * x_0**2 + 2 * alpha * x_nz * x_0 - alpha * x_nz ** 2
window_bounds = np.array(
    [
        current_time.to(current_time.u).m,
        x_nz.to(current_time.u).m,
        x_nz.to(current_time.u).m + 10,  # ensure flat after net zero
    ]
)

coeffs = np.array(
    [
        [-alpha.to(budget.u / current_time.u**2).m, 0.0],
        [(2 * alpha * (x_nz - current_time)).to(budget.u / current_time.u).m, 0.0],
        [
            (
                h
                - alpha * current_time**2
                + 2 * alpha * x_nz * current_time
                - alpha * x_nz**2
            )
            .to(budget.u)
            .m,
            h.to(budget.u).m,
        ],
    ]
)

# %%
piecewise_polynomial = scipy.interpolate.PPoly(
    c=coeffs,
    x=window_bounds,
    extrapolate=False,
)

# %%
cumulative_emms_quadratic_pathway = Timeseries(
    time=TimeAxis(
        values=np.hstack([current_time, x_nz]),
        value_last_bound=x_nz + Q(10, "yr"),
    ),
    continuous=TimeseriesContinuous(
        name="demo_quadratic_cumulative_emms",
        time_units=current_time.u,
        values_units=budget.u,
        piecewise_polynomial=piecewise_polynomial,
    ),
)
cumulative_emms_quadratic_pathway.plot(set_ylabel=True)

# %%
cumulative_emms_quadratic_pathway.differentiate().plot(set_ylabel=True)

# %%
cumulative_emms_quadratic_pathway.discrete

# %%
cumulative_emms_quadratic_pathway.time.bounds

# %%
yearly_steps = Q(
    np.arange(
        cumulative_emms_quadratic_pathway.time.bounds[0].to("yr").m,
        cumulative_emms_quadratic_pathway.time.bounds[-1].to("yr").m,
        Q(1, "yr").to("yr").m,
    ),
    "yr",
)

cumulative_emms_annual_step_pathway = Timeseries.from_arrays(
    all_values=cumulative_emms_quadratic_pathway.continuous.interpolate(yearly_steps),
    time_bounds=yearly_steps,
    interpolation=InterpolationOption.Linear,
    name="annual_steps",
)
cumulative_emms_annual_step_pathway.plot(set_ylabel=True)

# %%
cumulative_emms_annual_step_pathway.discrete

# %%
cumulative_emms_annual_step_pathway.differentiate().plot(set_ylabel=True)

# %%
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

continuous_kwargs = dict(alpha=0.7, linewidth=2)

cumulative_emms_quadratic_pathway.plot(
    ax=axes[0], continuous_kwargs=continuous_kwargs, set_ylabel=True
)
cumulative_emms_annual_step_pathway.plot(
    ax=axes[0], continuous_kwargs=continuous_kwargs
)

cumulative_emms_quadratic_pathway.differentiate().plot(
    ax=axes[1], continuous_kwargs=continuous_kwargs, set_ylabel=True
)
cumulative_emms_annual_step_pathway.differentiate().plot(
    ax=axes[1], continuous_kwargs=continuous_kwargs
)

for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

fig.tight_layout()
