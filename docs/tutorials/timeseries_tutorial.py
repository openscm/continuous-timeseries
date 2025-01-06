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
# # Timeseries
#
# Here we introduce our handling of timeseries.
# This builds on the introductions provided in
# [the discrete timeseries tutorial](../discrete_timeseries_tutorial)
# and [the continuous timeseries tutorial](../continuous_timeseries_tutorial).
# This general approach may be more unusual or unfamiliar
# to people used to working with arrays.
# This notebook is intended to serve as an introduction
# into one of the key concepts used in this package.
# For an overview into why the API is built like this, see
# [Why this API?](../../further-background/why-this-api).

# %% [markdown]
# ## Imports

# %%
import traceback

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint

import continuous_timeseries as ct
from continuous_timeseries.exceptions import ExtrapolationNotAllowedError

# %% [markdown]
# ## Set up pint

# %%
pint.set_application_registry(openscm_units.unit_registry)

# %% [markdown]
# ## Handy pint aliases

# %%
UR = pint.get_application_registry()
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
# ## The `Timeseries` class
#
# The `Timeseries` class is our representation of time series.

# %% [markdown]
# ### Initialising from arrays
#
# The most familiar/easiest way to initialise it is with its
# `from_arrays` class method.
# This takes in arrays which define the time series
# as well as an
# [`InterpolationOption`](../../api/continuous_timeseries/discrete_to_continuous/interpolation_option/#continuous_timeseries.discrete_to_continuous.interpolation_option.InterpolationOption).
# The interpolation option is key,
# because it allows us to take all the guesswork out of
# interpolation, extrapolation, integration and differentiateion.
# For more details on how this conversion is done, see
# [our docs on discrete to continuous conversions](../further-background/discrete_to_continuous_conversions).
#
# Let's assume that our arrays are the following.

# %%
time_axis = Q([1850.0, 1900.0, 1950.0, 2000.0, 2010.0, 2020.0, 2030.0, 2050.0], "yr")
values = Q([0.0, 10.0, 20.0, 50.0, 100.0, 100.0, 80.0, 60.0], "MtC / yr")

# %% [markdown]
# The interpolation option defines how to translate
# from the values in the arrays to a continuous timeseries.
# For example, let's assume
# that we want linear interpolation between our points.
# This can be achieved as shown below.

# %%
ts_linear = ct.Timeseries.from_arrays(
    x=time_axis,
    y=values,
    interpolation=ct.InterpolationOption.Linear,
    name="linear",
)
ts_linear.plot()

# %% [markdown]
# ### Unpacking the class
#
# So, what has happened here?
# The input arrays were taken and converted to a continuous representation.
# We also stored the time axis, for convenience when plotting
# and to make it possible to return to the discrete representation we started from.
# We can see the discrete form with the `discrete` property.

# %%
ts_linear.discrete

# %% [markdown]
# The full view of the timeseries shows the time axis
# and the continuous representation that was created.
# The continuous representation is an instance of the
# [`TimeseriesContinous`](../../api/continuous_timeseries/#continuous_timeseries.TimeseriesContinuous)
# class.

# %%
ts_linear

# %%
ts_linear.timeseries_continuous

# %% [markdown]
# By default, linear interpolation uses a
# [`ContinuousFunctionScipyPPoly`](../../api/continuous_timeseries/timeseries_continuous/#continuous_timeseries.timeseries_continuous.ContinuousFunctionScipyPPoly)
# class for the continuous representation
# (as shown below).
# However, any class which matches the
# [`ContinuousFunctionLike`](../../api/continuous_timeseries/timeseries_continuous/#continuous_timeseries.timeseries_continuous.ContinuousFunctionLike)
# interface could be used.
# Having said that, in general we expect most users
# to not need to worry about these details
# (although, if you really want to get into the details of interpolation,
# maybe have a look at
# [the higher-order interpolation tutorial](../../tutorials/higher_order_interpolation)
# to start).

# %%
ts_linear.timeseries_continuous.function

# %% [markdown]
# ### Other interpolation choices
#
# Linear interpolation is, of course, not the only choice.
# Here we show our other inbuilt interpolation options
# (as discussed previously,
# you can also supply your own continuous representations
# and there is more detail in
# [our docs on discrete to continuous conversions](../further-background/discrete_to_continuous_conversions)).

# %%
# Create a dictionary of `Timeseries` for easier re-use.
ts_interp = {}
for interp_option in (
    (ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed),
    (ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen),
    (ct.InterpolationOption.PiecewiseConstantNextLeftClosed),
    (ct.InterpolationOption.PiecewiseConstantNextLeftOpen),
    (ct.InterpolationOption.Linear),
    (ct.InterpolationOption.Cubic),
):
    ts_interp[interp_option] = ct.Timeseries.from_arrays(
        x=time_axis,
        y=values,
        interpolation=interp_option,
        name=interp_option.name,
    )

ts_interp.keys()

# %%
# Plot the options
fig, axs = plt.subplot_mosaic(
    [["piecewise_constant_previous"], ["piecewise_constant_next"], ["other"]],
    figsize=(12, 8),
)

discrete_points_plotted = []
for ts, ax, marker in (
    (
        ts_interp[ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed],
        axs["piecewise_constant_previous"],
        "^",
    ),
    (
        ts_interp[ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen],
        axs["piecewise_constant_previous"],
        "o",
    ),
    (
        ts_interp[ct.InterpolationOption.PiecewiseConstantNextLeftClosed],
        axs["piecewise_constant_next"],
        "^",
    ),
    (
        ts_interp[ct.InterpolationOption.PiecewiseConstantNextLeftOpen],
        axs["piecewise_constant_next"],
        "o",
    ),
    (ts_interp[ct.InterpolationOption.Linear], axs["other"], "^"),
    (ts_interp[ct.InterpolationOption.Cubic], axs["other"], "o"),
):
    if ax not in discrete_points_plotted:
        ax.scatter(
            time_axis,
            values,
            marker="x",
            s=130,
            label="Input discrete points",
            color="red",
            zorder=4,
        )
        discrete_points_plotted.append(ax)

    ts.plot(
        ax=ax,
        continuous_plot_kwargs=dict(
            alpha=0.7,
            label=f"{ts.name} interpolation",
        ),
        show_discrete=True,
        discrete_plot_kwargs=dict(
            label=f"{ts.name} values at boundaries",
            zorder=3,
            marker=marker,
        ),
    )

for ax in axs.values():
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()

fig.tight_layout()

# %% [markdown]
# ### Intepolation
#
# With our continuous representations, interpolation is trivial.

# %%
interp_times = Q([1950, 1960, 1970, 1980, 1990, 2000], "yr")

fig, ax = plt.subplots()

for interp_option, ts in ts_interp.items():
    ts.interpolate(interp_times).plot(
        ax=ax,
        show_discrete=True,
        continuous_plot_kwargs=dict(alpha=0.4, linestyle=":", label=""),
        discrete_plot_kwargs=dict(
            zorder=3,
            marker="x",
            alpha=0.7,
            s=130,
            label=f"{ts.name} interpolated",
        ),
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %% [markdown]
# If you don't explicitly allow it,
# you will get an error if you try and extrapolate.

# %%
ts_interp[ct.InterpolationOption.Linear].timeseries_continuous.domain

# %%
try:
    ts_interp[ct.InterpolationOption.Linear].interpolate(Q([2000, 2025, 2055], "yr"))
except ExtrapolationNotAllowedError:
    traceback.print_exc(limit=0)

# %% [markdown]
# With `allow_extrapolation=True`, you can also extrapolate.

# %%
extrap_times = Q(np.arange(2035, 2070 + 1, 10), "yr")

interp_colours = {
    ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed: "tab:blue",
    ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen: "tab:orange",
    ct.InterpolationOption.PiecewiseConstantNextLeftClosed: "tab:red",
    ct.InterpolationOption.PiecewiseConstantNextLeftOpen: "tab:green",
    ct.InterpolationOption.Linear: "tab:purple",
    ct.InterpolationOption.Cubic: "tab:olive",
}

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))

for interp_option, ts in ts_interp.items():
    ts.plot(
        ax=axes[0],
        continuous_plot_kwargs=dict(
            alpha=0.5, color=interp_colours[interp_option], linewidth=3
        ),
    )
    ts.plot(
        ax=axes[1],
        continuous_plot_kwargs=dict(
            color=interp_colours[interp_option], label="", zorder=3, linewidth=3
        ),
    )
    ts.interpolate(extrap_times, allow_extrapolation=True).plot(
        ax=axes[1],
        continuous_plot_kwargs=dict(
            alpha=0.7, linestyle="--", color=interp_colours[interp_option]
        ),
    )

for ax in axes:
    ax.set_xlim(extrap_times.min(), extrap_times.max())

axes[0].set_title("Raw")
axes[1].set_title("Extrapolated")
axes[1].legend()

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %% [markdown]
# ### Integration and differentiation
#
# With our continuous representations,
# integration and differentiation are also trivial.

# %%
continuous_plot_kwargs = dict(alpha=0.7, linestyle="-")

fig, axes_ar = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes_ar.flatten()

for ts_plot in ts_interp.values():
    ts_plot.plot(ax=axes[0], continuous_plot_kwargs=continuous_plot_kwargs)
    ts_plot.differentiate().plot(
        ax=axes[1],
        continuous_plot_kwargs={**continuous_plot_kwargs, "label": ts_plot.name},
    )

    integration_constant = Q(0, "GtC")
    integral = ts_plot.integrate(integration_constant=integration_constant)
    integral.plot(ax=axes[2], continuous_plot_kwargs=continuous_plot_kwargs)

    integral.integrate(integration_constant=Q(0.0, "GtC yr")).plot(
        ax=axes[3], continuous_plot_kwargs=continuous_plot_kwargs
    )

axes[0].set_title("Time series")
axes[0].yaxis.set_units(UR.Unit("MtCO2 / yr"))

axes[1].set_title("Derivative")
axes[1].yaxis.set_units(UR.Unit("MtC / yr^2"))

axes[2].set_title("Integral")
axes[2].yaxis.set_units(UR.Unit("GtC"))

axes[3].set_title("Double integral")
axes[3].yaxis.set_units(UR.Unit("TtC yr"))


axes[0].legend()

fig.tight_layout()

# %% [markdown]
# ### Integral-preserving interpolation
#
# One other useful method is `update_interpolation_integral_preserving`.
# This allows you to do integral-preserving interpolation,
# which is very helpful if you want to conserve carbon.
# For example, we can go from our linear representation
# to annual-average emissions as shown below.

# %%
ts_integral_preserving_demo_start = ct.Timeseries.from_arrays(
    x=Q([2025, 2030, 2040, 2050, 2060, 2100], "yr"),
    y=Q([10.0, 10.0, 5.0, 0.0, -2.0, 0.0], "GtC / yr"),
    interpolation=ct.InterpolationOption.Linear,
    name="integral_preserving_demo_start",
)

# %%
integral_preserving_demo_annual_time_axis = Q(
    np.arange(
        ts_integral_preserving_demo_start.time_axis.bounds.min().to("yr").m,
        ts_integral_preserving_demo_start.time_axis.bounds.max().to("yr").m + 0.1,
        1,
    ),
    "yr",
)
integral_preserving_demo_annual_time_axis

# %%
ts_linear_annual_to_show = ts_integral_preserving_demo_start.interpolate(
    integral_preserving_demo_annual_time_axis
)
annual_average = ts_linear_annual_to_show.update_interpolation_integral_preserving(
    interpolation=ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
    name_res="annual_average",
)

# %%
fig, axes = plt.subplots(nrows=2)

for ax in axes:
    ts_linear_annual_to_show.plot(ax=ax)
    annual_average.plot(ax=ax)

axes[0].legend()

axes[1].set_xlim([2030, 2040])
axes[1].set_ylim([5, 10])

fig.tight_layout()

# %% [markdown]
# If we want, we could also calculate integral-preserving average emissions
# over some custom time period.

# %%
decadal_average = ts_integral_preserving_demo_start.interpolate(
    Q(np.hstack([[2025, 2040], np.arange(2060, 2100 + 1, 20)]), "yr"),
).update_interpolation_integral_preserving(
    interpolation=ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
    name_res="custom_average",
)

# %%
fig, axes = plt.subplots(nrows=2, sharex=True)

ts_linear_annual_to_show.plot(ax=axes[0])
annual_average.plot(ax=axes[0])
decadal_average.plot(ax=axes[0])

ts_linear_annual_to_show.integrate(Q(0, "GtC")).plot(ax=axes[1])
annual_average.integrate(Q(0, "GtC")).plot(ax=axes[1])
decadal_average.integrate(Q(0, "GtC")).plot(ax=axes[1])

axes[0].set_title("Emissions")
axes[0].legend()
axes[1].set_title("Cumulative emissions")
axes[1].yaxis.set_units(UR.Unit("GtC"))

fig.tight_layout()

# %% [markdown]
# ### Plotting
#
# This class makes plotting simple, as seen above.

# %% [markdown]
# #### Resolution of the plot
#
# As far as we can tell, plotting continuous functions isn't trivial.
# So, we instead simply sample the function at many points,
# then plot using a straight line between points.
# In most cases, our eye can't tell the difference
# (and for linear interpolation, the choice of resolution makes no difference at all).
# However, in some cases it is useful to be able to control this resolution.
# We demonstrate how to do this below.

# %%
fig, axes_ar = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
axes = axes_ar.flatten()

for i, ts_plot in enumerate(ts_interp.values()):
    for res_increase in (1, 2, 5, 10, 100):
        ts_plot.plot(
            ax=axes[i],
            continuous_plot_kwargs=dict(
                res_increase=res_increase,
                label=f"{res_increase=}",
                alpha=0.7,
                linestyle="--",
            ),
        )

    axes[i].set_title(ts_plot.name)

axes[0].legend()

fig.tight_layout()

# %% [markdown]
# #### Customising plots and other features

# %% [markdown]
# By default, the plotted points
# are labelled with the name of the `Timeseries` object.
# This is shown if you add a legend to your plot.

# %%
ax = ts_linear.plot()
ax.legend()

# %% [markdown]
# The `show_continuous`, `continuous_plot_kwargs`,
# `show_discrete` and `discrete_plot_kwargs`
# arguments allow you to control how the plot is created.
# The `show*` arguments allow you to control which views
# of the timeseries are plotted and the `*kwargs`
# arguments allow you to pass arguments through to the relevant
# method of the underlying axes.
# This gives you full control over the plot.

# %%
fig, ax = plt.subplots()

y_unit = "MtC / year"
ax.set_ylabel(y_unit)
ax.yaxis.set_units(UR.Unit(y_unit))

ts_linear.plot(
    ax=ax,
    continuous_plot_kwargs=dict(
        color="tab:orange",
        label="demo",
        linestyle="--",
        linewidth=2,
    ),
)
ts_linear.plot(
    ax=ax,
    show_continuous=False,
    show_discrete=True,
    discrete_plot_kwargs=dict(
        marker="o",
        s=150,
        color="tab:orange",
        label="demo_discrete_only",
        zorder=3,
    ),
)
ts_interp[ct.InterpolationOption.Cubic].plot(
    ax=ax,
    continuous_plot_kwargs=dict(
        color="tab:blue",
        label="demo_continuous_and_discrete",
        linestyle=":",
    ),
    show_discrete=True,
    discrete_plot_kwargs=dict(
        alpha=0.7,
        zorder=3,
        marker="x",
        s=150,
        color="tab:blue",
        label="demo_continuous_and_discrete",
    ),
)

ax.grid()
ax.legend()
