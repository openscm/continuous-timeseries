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
# [or our docs on unit-aware plotting](../discrete_timeseries_tutorial#unit-aware-plotting).

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
# It is clear that we have to consider the size of the timesteps
# in order to integrate the emissions.
# So, we want an API that makes that easy.
#
# On top of this, the decision
# about whether to linearly interpolate between the emissions values
# or treat them as stepwise constant
# (i.e. assume that emissions are constant between the defining points)
# will have a big difference on the result,
# yet we do not have any information about what choice was intended based on the data.
# So, we want an API that solves this too.

# %% [markdown]
# ## The proposed solution

# %% [markdown]
# Our proposed API to solve this is the
# [`Timeseries`](../../api/continuous_timeseries/#continuous_timeseries.Timeseries)
# class, along with the associated
# [`TimeseriesContinous`](../../api/continuous_timeseries/#continuous_timeseries.TimeseriesContinuous)
# and
# [`TimeseriesDiscrete`](../../api/continuous_timeseries/#continuous_timeseries.TimeseriesDiscrete)
# classes.

# %% [markdown]
# ### Discrete representation
#
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
# ### Continuous representation
#
# To go to a continuous representation,
# we have to specify the interpolation option we want to use.
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

discrete_points_plotted = []
for interp_option, ax, marker in (
    (
        ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
        axs["piecewise_constant_previous"],
        "^",
    ),
    (
        ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen,
        axs["piecewise_constant_previous"],
        "o",
    ),
    (
        ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
        axs["piecewise_constant_next"],
        "^",
    ),
    (
        ct.InterpolationOption.PiecewiseConstantNextLeftOpen,
        axs["piecewise_constant_next"],
        "o",
    ),
    (ct.InterpolationOption.Linear, axs["other"], "x"),
):
    if ax not in discrete_points_plotted:
        covid_emissions.plot(
            ax=ax,
            marker="+",
            s=130,
            label="Input discrete points",
            color="red",
            zorder=3,
        )
        discrete_points_plotted.append(ax)

    continuous = covid_emissions.to_continuous_timeseries(interpolation=interp_option)
    continuous.plot(
        time_axis=covid_emissions.time_axis,
        ax=ax,
        alpha=0.7,
        label=f"{interp_option.name} interpolation",
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

# %% [markdown]
# These different interpolation options have an impact on integration too,
# which is often what we are most interested in when thinking about CO$_2$ emissions
# (cumulative CO$_2$ emissions are more useful than instantaneous in many cases).

# %%
fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(12, 8))

continuous_linear_integral_values = (
    covid_emissions.to_continuous_timeseries(
        interpolation=ct.InterpolationOption.Linear
    )
    .integrate(Q(0, "GtC"))
    .interpolate(covid_emissions.time_axis)
)

for interp_option in (
    (ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed),
    (ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen),
    (ct.InterpolationOption.PiecewiseConstantNextLeftClosed),
    (ct.InterpolationOption.PiecewiseConstantNextLeftOpen),
    (ct.InterpolationOption.Linear),
):
    continuous = covid_emissions.to_continuous_timeseries(interpolation=interp_option)
    continuous.plot(
        time_axis=covid_emissions.time_axis,
        ax=axes[0],
        alpha=0.7,
        label=interp_option.name,
    )

    integral = continuous.integrate(Q(0, "GtC"))
    integral.plot(
        time_axis=covid_emissions.time_axis,
        ax=axes[1],
        alpha=0.7,
        label=interp_option.name,
    )

    axes[2].plot(
        covid_emissions.time_axis.bounds,
        integral.interpolate(covid_emissions.time_axis)
        - continuous_linear_integral_values,
        label=interp_option.name,
    )

axes[0].set_title("Continuous")
axes[1].set_title("Integral")
axes[2].set_title("Difference from integral of linear interpolation")
axes[2].yaxis.set_units(UR.Unit("MtCO2"))
for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

fig.tight_layout()

# %% [markdown]
# ### Timeseries representation
#
# In general, users won't interact with
# [`TimeseriesContinous`](../../api/continuous_timeseries/#continuous_timeseries.TimeseriesContinuous)
# and [`TimeseriesDiscrete`](../../api/continuous_timeseries/#continuous_timeseries.TimeseriesDiscrete)
# directly.
# Instead, they will mostly use the
# [`Timeseries`](../../api/continuous_timeseries/#continuous_timeseries.Timeseries)
# class.
# This offers the same behaviour as above, with a simpler API.

# %%
ts = ct.Timeseries.from_arrays(
    values_at_bounds=emissions_covid,
    time_axis_bounds=years_covid,
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    name="covid_emissions",
)
ts

# %%
ts.plot()

# %%
ts.integrate(Q(100, "GtC")).plot()

# %% [markdown]
# This API also solves our integration issue.
# If we pass in a timeseries with time steps of different sizes,
# the integral is nonetheless evaluated correctly
# (making it easy to avoid one of the most common gotcha's
# in climate science/scenario development).

# %%
ts_varying_step_size = ct.Timeseries.from_arrays(
    values_at_bounds=Q([0.0, 0.2, 0.3, 1.2, 10.3], "GtCO2 / yr"),
    time_axis_bounds=Q([1750, 1800, 1825, 1900, 2000], "yr"),
    interpolation=ct.InterpolationOption.Linear,
    name="varying_step_size",
)
ts_varying_step_size

# %%
ax = ts_varying_step_size.plot()
ax.grid()

# %%
ax = ts_varying_step_size.integrate(Q(0, "GtC")).plot()
ax.grid()

# %% [markdown]
# The timeseries representation also makes it trivial
# to do integral-preserving interpolation
# on the time axis of your choice.

# %%
decadal_steps = ct.TimeAxis(
    np.arange(
        ts_varying_step_size.time_axis.bounds.m.min(),
        ts_varying_step_size.time_axis.bounds.m.max() + 0.01,
        10,
    )
    * ts_varying_step_size.time_axis.bounds.u
)

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
for label, updated_time_axis, updated_interpolation in (
    ("starting_point", None, None),
    (
        "same time - piecewise constant previous",
        ts_varying_step_size.time_axis,
        ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    ),
    (
        "decadal steps - piecewise constant previous",
        decadal_steps,
        ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    ),
):
    if updated_time_axis is not None:
        updated_time = ts_varying_step_size.update_time(updated_time_axis)
    else:
        updated_time = ts_varying_step_size

    if updated_interpolation is not None:
        updated_time_and_interp = updated_time.update_interpolation_integral_preserving(
            updated_interpolation
        )
    else:
        updated_time_and_interp = updated_time

    updated_time_and_interp.plot(ax=axes[0], continuous_plot_kwargs=dict(label=label))

    updated_time_and_interp.integrate(Q(0, "GtC")).plot(
        ax=axes[1], continuous_plot_kwargs=dict(label=label)
    )

axes[0].set_title("Emissions")
axes[1].set_title("Cumulative emissions")
for ax in axes:
    ax.grid()
    ax.legend()

fig.tight_layout()

# %% [markdown]
# ## Summary
#
# This notebook has explained why we have built this API.
# In short, it solves four key problems:
#
# 1. Removing the ambiguity about what to do between discrete time points
#    (which removes ambiguity about the value of integrals and derivatives).
# 1. Making interpolation, integration and differentation trivial,
#    even on time axes with uneven steps.
# 1. Making plotting in a way that actually represents the meaning of the data trivial.
# 1. Doing all of the above with unit awareness,
#    to remove that entire class of headaches and gotchas.
#
# Alongside solving these issues, it also has the added bonus
# that it makes it easy to integral-preserving interpolation,
# which can be very helpful for reporting over different time periods
# with different conventions.
