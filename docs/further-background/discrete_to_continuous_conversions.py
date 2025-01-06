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
# # Converting from discrete to continuous time series
#
# Here we explore the conversion
# from discrete to continuous time series in more detail.
#
# Our general advice is:
# if you want completely control over this conversion,
# you have to do it yourself, our pre-built solutions
# do not anticipate all use cases
# (for an example of how to create your own custom conversion, see
# [custom interpolation](../../tutorials/higher_order_interpolation#custom)
# ).
#
# With the general advice out of the way,
# let's dive into what is pre-built
# and available already within Continuous Timeseries.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint

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
# [or our docs on unit-aware plotting](../../tutorials/discrete_timeseries_tutorial#unit-aware-plotting).

# %%
UR.setup_matplotlib(enable=True)

# %% [markdown]
# ## In-built discrete to continuous conversions
#
# Below, we plot the results of the in-built conversions.
# For each conversion, we plot the input x- and y-points
# as well as the resulting continuous timeseries,
# including how it behaves when extrapolated.

# %%
x = Q([2010.0, 2015.0, 2020.0, 2030.0, 2050.0, 2075.0], "yr")
y = Q([1.0, 1.5, 3.0, 2.0, 2.0, 0.5], "MtCH4 / yr")

# %%
# Some other constants we'll want later
pre_step = Q(5, "yr")
extrapolate_pre = np.sort(x[0] - pre_step * np.arange(3))
extrapolate_pre

post_step = Q(5, "yr")
extrapolate_post = x[-1] + post_step * np.arange(3)
extrapolate_post

# %%
# Split the interpolation options
# to help with later explanation.
special_interps = (
    ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen,
    ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
)

standard_cases = [io for io in ct.InterpolationOption if io not in special_interps]
edge_cases = [io for io in ct.InterpolationOption if io in special_interps]


# %%
# Helper functions
def get_mosaic(elements, plots_per_row):  # noqa:D103
    mosaic = []
    for i in range(len(elements)):
        if (i + 1) % plots_per_row < 1:
            mosaic.append(elements[i + 1 - plots_per_row : i + 1])

    leftovers = (i + 1) % plots_per_row
    if leftovers > 0:
        mosaic.append([elements[-leftovers:]])
        for _ in range(plots_per_row - leftovers):
            mosaic[-1].append("")

    return mosaic


def plot_interpolation_options(interp_options, axs):  # noqa:D103
    for interp_option in interp_options:
        ts = ct.Timeseries.from_arrays(
            x=x,
            y=y,
            interpolation=interp_option,
            name=repr(interp_option),
        )
        cpk_common = dict(
            alpha=0.7,
            res_increase=100,
            # Just show markers to avoid
            # matplotlib line joining
            # giving the wrong impression
            marker="o",
            markersize=3,
            linewidth=0,
        )

        axs[interp_option].scatter(
            x,
            y,
            label="Input x- and y-points",
            marker="+",
            s=150,
            linewidth=3,
            color="tab:orange",
            zorder=3,
        )

        axs[interp_option].scatter(
            x,
            ts.timeseries_continuous.interpolate(x),
            label="Continuous value at input x-points",
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            s=150,
        )

        ts.plot(
            ax=axs[interp_option],
            continuous_plot_kwargs=dict(
                label="Result from interpolation",
                color="tab:blue",
                **cpk_common,
            ),
        )

        for label, times, colour in (
            ("Extrapolation pre-domain", extrapolate_pre, "tab:green"),
            ("Extrapolation post-domain", extrapolate_post, "tab:red"),
        ):
            ts.interpolate(times, allow_extrapolation=True).plot(
                ax=axs[interp_option],
                continuous_plot_kwargs=dict(label=label, color=colour, **cpk_common),
            )

        axs[interp_option].set_title(f"{interp_option.name} ({interp_option.value})")
        axs[interp_option].legend()


# %% [markdown]
# ### Standard cases
#
# These are the standard cases.
# For these cases, things more or less do what you'd expect.
# In particular, the continuous representation
# has the same y-value at each input x-point
# as the input y-value.

# %%
interpolation_options = standard_cases
mosaic = get_mosaic(interpolation_options, 2)
fig, axs = plt.subplot_mosaic(mosaic, figsize=(14, 15))
plot_interpolation_options(interpolation_options, axs)

# %% [markdown]
# ### Edge cases
#
# For the following interpolation types,
# things can be a bit more confusing.
# For these cases, the continuous representation
# can have a different y-value at each input x-point
# to the input y-value.
# This interpolation choice might be useful
# in some cases, but it can be confusing
# so please use it with caution.

# %%
interpolation_options = edge_cases
mosaic = get_mosaic(interpolation_options, 2)
fig, axs = plt.subplot_mosaic(mosaic, figsize=(14, 5))
plot_interpolation_options(interpolation_options, axs)

# %% [markdown]
# To help users, if you inadvertently do such a conversion,
# you will get a warning.

# %%
ct.TimeseriesDiscrete(
    name="example",
    time_axis=ct.TimeAxis(Q([10, 11, 12], "month")),
    values_at_bounds=ct.ValuesAtBounds(Q([1, 2, 3], "kg")),
).to_continuous_timeseries(edge_cases[0])

# %%
# In contrast, this does not raise a warning

# %%
ct.TimeseriesDiscrete(
    name="example",
    time_axis=ct.TimeAxis(Q([10, 11, 12], "month")),
    values_at_bounds=ct.ValuesAtBounds(Q([1, 2, 3], "kg")),
).to_continuous_timeseries(standard_cases[0])

# %%
# As a reminder, these are the y-values
y

# %%
# Compare the above to the values_at_bounds below
# (look closely at the last value)
ct.Timeseries.from_arrays(
    x=x, y=y, name="demo", interpolation=ct.InterpolationOption.Linear
).update_interpolation(
    ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen
).discrete
