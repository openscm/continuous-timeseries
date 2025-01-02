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

import continuous_timeseries as ct

# %% [markdown]
# ## Handy pint aliases

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %% [markdown]
# ## Set up matplotlib to work with pint
#
# For details, see the pint docs
# ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
# [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))
# [or our docs on unit-aware plotting](../discrete_timeseries_tutorial#unit-aware-plotting).  # noqa: E501

# %%
UR.setup_matplotlib(enable=True)

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
ax.plot(years_covid.m, emissions_covid.m, marker="o")
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
x_vals = np.hstack([years_covid, years_covid[-1] + Q(1, "yr")]).m
y_vals = np.hstack([emissions_covid, emissions_covid[-1]]).m
ax.step(x_vals, y_vals, where="post", marker="o")
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
# Our proposed API to solve this is the
# [`Timeseries`][continuous_timeseries.Timeseries] class,
# along with the associated
# [`TimeseriesContinous`][continuous_timeseries.TimeseriesContinous]
# and [`TimeseriesDiscrete`][continuous_timeseries.TimeseriesDiscrete] classes.

# %% [markdown]
# In the case of our COVID emissions example,
# we would capture the timeseries in its discrete form as shown below.

# %%
covid_emissions = ct.TimeseriesDiscrete(
    name="co2_emissions",
    time_axis=ct.TimeAxis(years_covid),
    values_at_bounds=ct.ValuesAtBounds(emissions_covid),
)
covid_emissions


# %% [markdown]
# To go to a continuous representation, we have to specify the interpolation option we want to use.
# For emissions, the most accurate choice is generally piecewise constant,
# because emissions are typically reported as total emissions for the year
# (i.e. the number is the average over the year,
# which isn't well captured by linear interpolation,
# at least not naively).

# %%
covid_emissions_linear_interpolation = covid_emissions.to_continuous_timeseries(
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed
)
covid_emissions_linear_interpolation.plot(time_axis=covid_emissions.time_axis)
covid_emissions_linear_interpolation

# %% [markdown]
# We visualise the implications of different interpolation options below.


# %%
fig, axs = plt.subplot_mosaic(
    [["piecewise_constant_previous"], ["piecewise_constant_next"], ["other"]],
    figsize=(12, 8),
)

for interp_option, ax, marker in (
    (
        ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        axs["piecewise_constant_previous"],
        "x",
    ),
    # (ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen, "o"),
    (
        ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
        axs["piecewise_constant_next"],
        "x",
    ),
    # (ct.InterpolationOption.PiecewiseConstantNextLeftOpen, "o"),
    (ct.InterpolationOption.Linear, axs["other"], "x"),
):
    continuous = covid_emissions.to_continuous_timeseries(interpolation=interp_option)
    continuous.plot(
        time_axis=covid_emissions.time_axis,
        ax=ax,
        alpha=0.7,
        label=interp_option.name,
    )
    ax.scatter(
        covid_emissions.time_axis.bounds,
        continuous.interpolate(covid_emissions.time_axis),
        marker=marker,
        s=130,
        alpha=0.4,
        label=f"{interp_option.name} at boundaries",
    )

for ax in axs.values():
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

fig.tight_layout()

# %%
assert False, "add other interpolation choices"

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
# These different interpolation options have a clearer impact once we integrate.

# %%
assert False, "consider plotting delta in integral"

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
assert False, "Make this to a how-to too and write something like 'This also allows you to create sharp step in a forcing function, something you can't do with discrete timeseries"

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
import copy


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

    # Get rid of this, put it in the 'budget' module
    def to_annual_piecewise_constant_integral_preserving(
        self,
        name_res: str | None = None,
    ):
        res = (
            self.update_time_to_annual_steps().update_interpolation_integral_preserving(
                interpolation=InterpolationOption.PiecewiseConstantNextLeftClosed,
                name_res=name_res,
            )
        )

        return res

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

    def update_interpolation(self, interpolation: InterpolationOption):
        continuous = discrete_to_continuous(
            discrete=self.discrete,
            interpolation=interpolation,
        )

        return type(self)(
            time=self.time,
            continuous=continuous,
        )

    def update_interpolation_integral_preserving(
        self,
        interpolation: InterpolationOption,
        name_res: str | None = None,
    ):
        # Doesn't matter as will be lost when we differentiate
        integration_constant = Q(
            0.0, self.continuous.values_units * self.continuous.time_units
        )

        if interpolation in (
            InterpolationOption.PiecewiseConstantPreviousLeftClosed,
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            InterpolationOption.PiecewiseConstantNextLeftClosed,
            InterpolationOption.PiecewiseConstantNextLeftOpen,
        ):
            interpolation_cumulative = InterpolationOption.Linear

        elif interpolation in (InterpolationOption.Linear):
            interpolation_cumulative = InterpolationOption.Quadratic

        elif interpolation in (InterpolationOption.Quadratic):
            interpolation_cumulative = InterpolationOption.Cubic

        else:
            raise NotImplementedError(interpolation)

        res = (
            self.integrate(integration_constant)
            .update_interpolation(interpolation_cumulative)
            .differentiate(name_res=name_res)
        )

        return res

    def update_time(self, time: TimeAxis):
        # Should check here that times are compatible with extrapolation choices
        return type(self)(
            time=time,
            continuous=self.continuous,
        )

    # Get rid of this, put it in the 'budget' module
    def update_time_to_annual_steps(self):
        # TODO (?): allow extrapolation choices here too
        #           to allow people to go forward/backward one step

        if self.time.bounds[0].to("yr").m % 1.0 == 0.0:
            # If the first value is a year value, we can simply use it
            arange_first_val = self.time.bounds[0].to("yr").m
        else:
            # Round up
            arange_first_val = np.ceil(self.time.bounds[0].to("yr").m)

        if self.time.bounds[-1].to("yr").m % 1.0 == 0.0:
            # If the last value is a year value, we can include it in the output
            arange_last_val = self.time.bounds[-1].to("yr").m + 0.1
        else:
            arange_last_val = self.time.bounds[-1].to("yr").m

        yearly_steps = (
            np.arange(
                arange_first_val,
                arange_last_val,
                1.0,
            )
            * self.time.values.to("yr").u
        )
        annual_time_axis = TimeAxis.from_bounds(yearly_steps)

        res = self.update_time(annual_time_axis)

        return res

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
                # Avoid mangling the input
                continuous_kwargs = copy.deepcopy(continuous_kwargs)
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
assert False, "split below here into budget docs and timeseries docs"

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
budget = Q(13.2, "GtC")

# %%
# We're solving for emissions, y
# as a function of time, x.
# Assume linear emissions
#     y(x) = e_0 * (1 - (x - x_0) / (x_nz - x_0))
#
# Simplifying slightly
#     y(x) = e_0 * (x_nz - x) / (x_nz - x_0)
#
# where e_0 is emissions at the known time (normally today), x_0,
# and x_nz is the net-zero time.
#
# By geometry, the integral of this curve between x_0 and x_nz is:
#
#     (x_0 - x_nz) * e_0 / 2
#
# You can also do this with calculus:
#
#     int_{x_0}^{x_nz} y(x) dx = int_{x_0}^{x_nz} e_0 * (x_nz - x) / (x_nz - x_0) x
#                              = [-e_0 (x_nz - x)**2 / (2 * (x_nz - x_0))]_{x_0}^{x_nz}
#                              = -e_0 (x_nz - x_0)**2 / (2 * (x_nz - x_0))) - e_0 (x_nz - x_nz)**2 / (2 * (x_nz - x_0)))
#                              = e_0 (x_0 - x_nz) / 2
#
# This integral should be equal to the allowed buget:
#
#     (x_0 - x_nz) * e_0 / 2 = budget
#
#     therefore
#     x_nz = x_0 + 2 * budget / e_0
x_nz = current_time + 2 * budget / current_emms
x_nz

# %%
# Convert into coefficients for our polynomial
# by simply recalling the definition we started with
#    y(x) = e_0 * (1 - (x - x_0) / (x_nz - x_0))
#    y(x) = e_0 - e_0 * (x - x_0) / (x_nz - x_0)
#
# Therefore, coefficients in y = m * (x - x_0) + c are:
#    c = e_0
#    m = - e_0 / (x_nz - x_0)

extend_post_nz = Q(
    3.0, current_time.to("yr").u
)  # good default as it makes most splines behave

time_bounds = np.hstack(
    [
        current_time,
        x_nz,
        x_nz + extend_post_nz,  # ensure flat after net zero
    ]
)
window_bounds = time_bounds.to(current_time.u).m

coeffs = np.array(
    [
        [
            (-current_emms / (x_nz - current_time))
            .to(current_emms.u / current_time.u)
            .m,
            0.0,
        ],
        [current_emms.m, 0.0],
    ]
)

# %%
piecewise_polynomial = scipy.interpolate.PPoly(
    c=coeffs,
    x=window_bounds,
    extrapolate=False,
)

# %%
emms_linear_pathway = Timeseries(
    time=TimeAxis.from_bounds(time_bounds),
    continuous=TimeseriesContinuous(
        name="linear_emissions",
        time_units=current_time.u,
        values_units=current_emms.u,
        piecewise_polynomial=piecewise_polynomial,
    ),
)

# %%
fig, ax = plt.subplots()

emms_linear_pathway.plot(set_ylabel=True, ax=ax)
emms_linear_pathway.to_annual_piecewise_constant_integral_preserving(
    name_res="annual_average_equivalent"
).plot(ax=ax)
emms_linear_pathway.update_interpolation_integral_preserving(
    InterpolationOption.PiecewiseConstantNextLeftClosed,
    name_res="forgot_to_convert_to_annual",
).plot(ax=ax)
emms_linear_pathway.update_time_to_annual_steps().update_interpolation_integral_preserving(
    InterpolationOption.Linear, name_res="annual_then_linear"
).plot(ax=ax)

ax.legend()
ax.grid()

fig.tight_layout()

# %%
cumulative_emms_linear_pathway = (
    emms_linear_pathway.integrate(Q(0, "GtC"))
    .update_time_to_annual_steps()
    .update_interpolation(InterpolationOption.Linear)
)
cumulative_emms_linear_pathway.continuous.name = (
    "linear_cumulative_emissions_annual_time_step"
)

# %%
cumulative_emms_cubic_pathway = (
    emms_linear_pathway.integrate(Q(0, "GtC"))
    .update_time_to_annual_steps()
    .update_interpolation(InterpolationOption.Cubic)
)
cumulative_emms_cubic_pathway.continuous.name = (
    "cubic_cumulative_emissions_annual_time_step"
)

# %%
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

continuous_kwargs = dict(alpha=0.7, linewidth=2)

emms_linear_pathway.plot(
    ax=axes[0], continuous_kwargs=continuous_kwargs, set_ylabel=True
)
emms_linear_pathway.integrate(Q(0, "GtC")).plot(
    ax=axes[1], continuous_kwargs=continuous_kwargs, set_ylabel=True
)

cumulative_emms_linear_pathway.plot(ax=axes[1], continuous_kwargs=continuous_kwargs)
cumulative_emms_linear_pathway.differentiate().plot(
    ax=axes[0], continuous_kwargs=continuous_kwargs
)

cumulative_emms_cubic_pathway.plot(ax=axes[1], continuous_kwargs=continuous_kwargs)
cumulative_emms_cubic_pathway.differentiate().plot(
    ax=axes[0], continuous_kwargs=continuous_kwargs
)

for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

fig.tight_layout()
