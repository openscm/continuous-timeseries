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
# # Higher-order interpolation
#
# Here we discuss some of the perils of higher-order interpolation.
# Here, when we say 'higher-order interpolation',
# we mean interpolating with functions that are second-order and higher.
# We also show some of our tips for navigating higher-order interpolation.

# %% [markdown]
# ## Imports

# %%

import matplotlib.pyplot as plt
import openscm_units
import pint
import scipy.interpolate

import continuous_timeseries as ct

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
# ## Sample data

# %% [markdown]
# We start with a basic timeseries.
# It doesn't really matter what this represents,
# but it it is (very) approximately a potential pathway of CO$_2$ emissions.

# %%
time_points = Q([1750, 1800, 1850, 1900, 2000, 2030, 2050, 2100], "yr")
emissions = Q([0.0, 0.5, 1.3, 3.0, 6.0, 10.0, 0.0, 0.0], "GtC / yr")

# %%
fig, ax = plt.subplots()

ax.scatter(time_points, emissions)

ax.grid()

# %% [markdown]
# ## Interpolation

# %% [markdown]
# ### Naive
#
# We start by naively interpolating these emissions
# using the default options provided by Continuous Timeseries.

# %%
fig, axs = plt.subplot_mosaic(
    [["constant", "linear"], ["quadratic", "cubic"]], figsize=(12, 8)
)

already_plotted_discrete = []
for interp_option, ax in (
    (ct.InterpolationOption.PiecewiseConstantNextLeftClosed, axs["constant"]),
    (ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed, axs["constant"]),
    (ct.InterpolationOption.Linear, axs["linear"]),
    (ct.InterpolationOption.Quadratic, axs["quadratic"]),
    (ct.InterpolationOption.Cubic, axs["cubic"]),
):
    if ax not in already_plotted_discrete:
        ax.scatter(
            time_points,
            emissions,
            label="Discrete starting points",
            color="tab:gray",
            zorder=3,
        )
        already_plotted_discrete.append(ax)

    ts = ct.Timeseries.from_arrays(
        x=time_points,
        y=emissions,
        interpolation=interp_option,
        name="co2_emissions",
    )
    ts.plot(
        ax=ax,
        continuous_plot_kwargs=dict(label=interp_option.name, alpha=0.6, linewidth=2),
    )

for ax_name, ax in axs.items():
    ax.grid()
    ax.set_title(ax_name)
    ax.legend()

fig.tight_layout()

# %% [markdown]
# As you can see, all of these naive interpolation options have issues.
# The constant and linear interpolations are too simple.
# The quadratic and cubic interpolations fly off to
# undesirably negative levels.

# %% [markdown]
# ### Custom
#
# In general, interpolation isn't a trivial problem
# (if you know of a good discussion of this we can use as a source,
# please
# [open an issue](https://github.com/openscm/continuous-timeseries/issues/new?assignees=&labels=triage&projects=&template=default.md&title=Interpolation%20discussion%20source)).
# Hence, there is no easy answer to this.
#
# Having said that, Continuous Timeseries
# does support injecting your own continuous representation.
# Here we show how this can be done using
# [scipy's CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html).

# %%
cubic_spline = scipy.interpolate.CubicSpline(
    x=time_points.m,
    y=emissions.m,
    # Scipy's docs on this are very helpful.
    # Here, we are basically saying,
    # make the first derivative at either boundary equal to zero.
    # For our data, this makes sense.
    # For other data, a different choice chould be better
    # (e.g. we often use `bc_type=((1, 0.0), "not-a-knot")`
    # which means, have a first derivative of zero on the left,
    # on the right,
    # just use the same polynomial for the last two time steps).)
    bc_type=((1, 0.0), (1, 0.0)),
)

custom = ct.Timeseries(
    time_axis=ct.TimeAxis(time_points),
    timeseries_continuous=ct.TimeseriesContinuous(
        name="custom_spline",
        time_units=time_points.u,
        values_units=emissions.u,
        function=ct.timeseries_continuous.ContinuousFunctionScipyPPoly(cubic_spline),
        domain=(time_points.min(), time_points.max()),
    ),
)
custom

# %% [markdown]
# As you can see, this results in much less overshoot
# and might be a better choice of fit.

# %%
fig, axs = plt.subplot_mosaic(
    [["full", "full"], ["early_zoom", "late_zoom"]], figsize=(8, 6)
)

for i, (ax, xlim, ylim) in enumerate(
    zip(
        axs.values(),
        ((1750, 2100), (1750, 1760), (2090, 2100)),
        ((-14, 14), (-0.05, 0.3), (-5, 1)),
    )
):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.scatter(
        time_points,
        emissions,
        label="Discrete starting points",
        color="tab:gray",
        zorder=3,
    )

    continuous_plot_kwargs = dict(alpha=0.6, linewidth=2)

    ct.Timeseries.from_arrays(
        x=time_points,
        y=emissions,
        interpolation=ct.InterpolationOption.Cubic,
        name="naive_cubic",
    ).plot(ax=ax, continuous_plot_kwargs=continuous_plot_kwargs)

    custom.plot(ax=ax, continuous_plot_kwargs=continuous_plot_kwargs)

    ax.grid()

axs["full"].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

fig.tight_layout()

# %% [markdown]
# ## Conclusion
#
# Interpolation isn't a trivial problem in general,
# particularly quadratic or higher-order interpolation.
# However, Continuous Timeseries provides
# relatively straight-forward support for custom interpolation.
# This can be done using existing packages.
# If you want to define your own, completely custom interpolation,
# then the interface you need to match is defined by
# [`ContinuousFunctionLike`](../../api/continuous_timeseries/timeseries_continuous/#continuous_timeseries.timeseries_continuous.ContinuousFunctionLike).
