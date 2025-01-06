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
# # How to make a step forcing
#
# Here we explain how to make a sharp step forcing.

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
# Imagine you want to run an experiment with a step forcing.
# For example, forcing before 1850 is 0 W / m<sup>2</sup>
# while forcing from 1850 (inclusive) onwards
# is equal to 2 W / m<sup>2</sup>.

# %% [markdown]
# ## The naive solution
#
# One way you could achieve this
# is to simply use a timeseries with a really fine time axis.

# %%
time_fine = Q(np.linspace(1750, 2100, int(1e5)), "yr")
time_fine

# %%
forcing_fine = np.zeros_like(time_fine) * UR.Unit("W / m^2")
transition_year = 1850
forcing_fine[time_fine.m >= transition_year] = Q(2.0, "W / m^2")
forcing_fine

# %% [markdown]
# This works, but if you zoom in far enough,
# you see the discrete steps
# i.e. the edge is no longer sharp.
# In addition, you have to carry around an array
# with heaps of time points
# and whatever API you pass this to
# has to know how to deal with a fine array of points.

# %%
fig, axes = plt.subplots(ncols=4, sharey=True, figsize=(12, 4))

for i, xlim in enumerate(
    ((1750, 2100), (1849, 1851), (1849.9, 1850.1), (1849.99, 1850.01))
):
    axes[i].plot(time_fine, forcing_fine)
    axes[i].set_xlim(xlim)
    axes[i].grid()

fig.tight_layout()

# %% [markdown]
# ## The Continuous Timeseries solution
#
# With continuous timeseries, we can achieve the same thing
# with far fewer data points.

# %%
ts = ct.Timeseries.from_arrays(
    y=Q([0.0, 2.0, 2.0], "W / m^2"),
    x=Q([1750, 1850, 3000], "yr"),
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    name="step_forcing",
)
ts

# %% [markdown]
# This solution also maintains the sharp edge, no matter how far we zoom in.

# %%
fig, axes = plt.subplots(ncols=4, sharey=True, figsize=(12, 4))

for i, xlim in enumerate(
    ((1750, 2100), (1849, 1851), (1849.9, 1850.1), (1849.99, 1850.01))
):
    ts.plot(
        ax=axes[i],
        # Make sure that the plot resolution is high enough
        # to show what is actually going on.
        continuous_plot_kwargs=dict(res_increase=int(3e6)),
    )
    axes[i].set_xlim(xlim)
    axes[i].grid()

fig.tight_layout()

# %%
# No matter how fine we make the time axis,
# the sharp transition is maintained.
ts.interpolate(Q([1850 - 1e-9, 1850.0, 1850 + 1e-9], "yr")).discrete

# %% [markdown]
# ## A few other notes
#
# The way `ts` is created above, at 1850, the value is 2.0 W / m<sup>2</sup>.
# If you want the value at 1850 to be zero instead,
# you can use `PPolyPiecewiseConstantPreviousLeftOpen` interpolation
# as shown below.

# %%
ts_open = ct.Timeseries.from_arrays(
    x=Q([1750, 1850, 3000], "yr"),
    y=Q([0.0, 2.0, 2.0], "W / m^2"),
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftOpen,
    name="step_forcing",
)
ts_open.plot()

# %%
# Notice here that the 1850 value is 0.0 W / m^2, not 2.0 W / m^2
ts_open.discrete

# %% [markdown]
# You can also achieve something similar using
# `PPolyPiecewiseConstantNextLeftClosed`
# and `PPolyPiecewiseConstantNextLeftOpen`.
# The process is slightly different but the outcome
# is the largely the same,
# with the main difference being in interpolation
# (for a further overview of the differences,
# see the docs of
# [`InterpolationOption`](../../api/continuous_timeseries/discrete_to_continuous/interpolation_option/#continuous_timeseries.discrete_to_continuous.interpolation_option.InterpolationOption)).

# %%
ts_next_left_closed = ct.Timeseries.from_arrays(
    x=Q([1750, 1850, 3000], "yr"),
    y=Q([0.0, 0.0, 2.0], "W / m^2"),
    interpolation=ct.InterpolationOption.PiecewiseConstantNextLeftClosed,
    name="next_left_closed",
)
ts_next_left_closed.plot()

# %%
ts_next_left_closed.discrete

# %%
ts_next_left_open = ct.Timeseries.from_arrays(
    x=Q([1750, 1850, 3000], "yr"),
    y=Q([0.0, 0.0, 2.0], "W / m^2"),
    interpolation=ct.InterpolationOption.PiecewiseConstantNextLeftOpen,
    name="next_left_open",
)
ts_next_left_open.plot()

# %%
ts_next_left_open.discrete
