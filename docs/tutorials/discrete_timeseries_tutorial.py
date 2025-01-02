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
# # Discrete timeseries
#
# Here we introduce our handling of discrete timeseries.
# This part is not that unusual or unfamiliar,
# but it might be a helpful introduction
# into some of the concepts used in this package.

# %% [markdown]
# ## Imports

# %%
import traceback

import matplotlib.pyplot as plt
import matplotlib.units
import pint

from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.values_at_bounds import ValuesAtBounds

# %% [markdown]
# ## Handy pint aliases

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %% [markdown]
# ## The `TimeseriesDiscrete` class

# %% [markdown]
# ### Time axis
#
# The first thing we need to define our timeseries is a time axis.
# This is handled by the `TimeAxis` class.
# This class expects to be passed the bounds of the time steps.
# In other words, the first time step runs from `bounds[0]`, to `bounds[1]`,
# the second from `bounds[1]`, to `bounds[2]`,
# the third from `bounds[2]`, to `bounds[3]` etc.

# %%
time_axis = TimeAxis(
    Q([1900.0, 1950.0, 1975.0, 2000.0, 2010.0, 2020.0, 2030.0, 2050.0], "yr")
)
time_axis

# %% [markdown]
# ### Values
#
# The second thing we need is the values.
# These must represent the values at each bound (i.e. value) in `time_axis`.

# %%
values_at_bounds = ValuesAtBounds(
    Q([2.3, 4.5, 6.4, 10.0, 11.0, 12.3, 10.2, 3.5], "Gt / yr")
)
values_at_bounds

# %% [markdown]
# ### Initialisation
#
# Having created our values and time axis, we can now add a name
# and initialise our instance.

# %%
ts = TimeseriesDiscrete(
    name="example",
    time_axis=time_axis,
    values_at_bounds=values_at_bounds,
)
ts

# %% [markdown]
# ### Plotting
#
# It is trivial to plot with this class.

# %%
ts.plot()

# %% [markdown]
# One thing which you notice straight away is
# that the data is plotted as a scatter plot.
# This is deliberate: the data is discrete i.e. we have no information
# about what happens between the provided data points.
# This way of plotting makes that as clear and obvious as possible.

# %% [markdown]
# The other thing that is clear is the warning about the units.
# If you need a quick plot but don't want this warning,
# it can be disabled as shown below.

# %%
ts.plot(warn_if_plotting_magnitudes=False)

# %% [markdown]
# #### Unit-aware plotting
#
# However, better than this is to use unit-aware plotting.
# This can be done with pint
# ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
# [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html)).

# %% [markdown]
# Firstly, set-up matplotlib to use the unit registry.

# %%
UR.setup_matplotlib(enable=True)

# %% [markdown]
# If you just plot now, the units will appear on the output axis.

# %%
ts.plot()

# %% [markdown]
# This has the added benefit that timeseries in different, but compatible units,
# will automatically be plotted with the same units.

# %%
ts_compatible_unit = TimeseriesDiscrete(
    name="example_compatible_unit",
    time_axis=TimeAxis(
        Q(
            [(1932 + 6) * 12.0, 2000 * 12.0, (2010 + 5) * 12.0, (2025 + 3) * 12.0],
            "month",
        )
    ),
    values_at_bounds=ValuesAtBounds(Q([140.0, 160.0, 120.0, -10.0], "Mt / month")),
)

ax = ts.plot()
ts_compatible_unit.plot(ax=ax)

# %% [markdown]
# You can set the desired units too.

# %%
fig, ax = plt.subplots()

# Note: the auto-scaling can do some funny things
# depending on the order of operations.
# You may need to set the axis limits
# or labels by hand before/after plotting to make things look sensible.
x_unit = "month"
ax.set_xlabel(x_unit)
ax.xaxis.set_units(x_unit)

y_unit = "Mt / month"
ax.set_ylabel(y_unit)
ax.yaxis.set_units(y_unit)

ts.plot(ax=ax)
ts_compatible_unit.plot(ax=ax)

ax.legend()

# %% [markdown]
# Another benefit is that you won't be able to inadverantly plot timeseries
# with incompatible units on the same axes.
# Instead, you will get an error.

# %%
ts_incompatible_unit = TimeseriesDiscrete(
    name="example_incompatible_unit",
    time_axis=TimeAxis(
        Q(
            [(1932 + 6) * 12.0, 2000 * 12.0, (2010 + 5) * 12.0, (2025 + 3) * 12.0],
            "month",
        )
    ),
    values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 30.0, 40.0], "Mt")),
)

ax = ts.plot()
try:
    ts_incompatible_unit.plot(ax=ax)
except matplotlib.units.ConversionError:
    traceback.print_exc(limit=0)

# %% [markdown]
# Of course, you can always disable the unit-aware plotting if you really want.

# %%
UR.setup_matplotlib(enable=False)

fig, ax = plt.subplots()

ts.plot(ax=ax, warn_if_plotting_magnitudes=False)
ts_compatible_unit.plot(ax=ax, warn_if_plotting_magnitudes=False)
ts_incompatible_unit.plot(ax=ax, warn_if_plotting_magnitudes=False)

ax.legend()

# %% [markdown]
# #### Customising plots and other features

# %%
# Re-enable unit awareness for the rest of the notebook
UR.setup_matplotlib(enable=True)

# %% [markdown]
# By default, the plotted points
# are labelled with the name of the `TimeseriesDiscrete` object.
# This is shown if you add a legend to your plot.

# %%
ax = ts.plot()
ax.legend()

# %% [markdown]
# The `label` argument and any unrecognised arguments
# are simply passed through to the `scatter` method of the underlying axes.
# This gives you full control over the plot.

# %%
fig, ax = plt.subplots()

y_unit = "Gt / year"
ax.set_ylabel(y_unit)
ax.yaxis.set_units(y_unit)

ts.plot(ax=ax, marker="x", color="tab:green", label="demo")
ts_compatible_unit.plot(ax=ax, marker="o", color="tab:red", label="other timeseries")
ax.grid()
ax.legend()
