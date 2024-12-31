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
import matplotlib.pyplot as plt
import pint

from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.values_at_bounds import ValuesAtBounds

# %% [markdown]
# ## Set up pint

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
# By default, the plotted points
# are labelled with the name of the `TimeseriesDiscrete` object.
# This is shown if you add a legend to your plot.

# %%
ax = ts.plot()
ax.legend()

# %% [markdown]
# There are then other things that can be added to make the plot slightly nicer.
# For example, axis labels in line with the units of the data.

# %%
ts.plot(set_xlabel=True, set_ylabel=True)

# %% [markdown]
# You can also specify the units in which you want to plot the data.

# %%
ts.plot(x_units="months", y_units="Mt / yr", set_xlabel=True, set_ylabel=True)

# %% [markdown]
# Any unrecognised arguments
# are simply passed through to the `scatter` method of the underlying axes.
# This gives you full control over the plot.

# %%
fig, ax = plt.subplots()

ax = ts.plot(
    ax=ax, set_xlabel=True, set_ylabel=True, marker="x", color="tab:green", label="demo"
)
ax.grid()
ax.legend()
