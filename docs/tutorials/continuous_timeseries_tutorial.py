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
# # Continuous timeseries
#
# Here we introduce our handling of continuous timeseries.
# This part may be more unusual or unfamiliar
# to people used to working with arrays,
# so it serves as an introduction
# into some of the concepts used in this package.

# %% [markdown]
# ## Imports

# %%
import traceback

import matplotlib.pyplot as plt
import matplotlib.units
import numpy as np
import pint
import scipy.interpolate

from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
from continuous_timeseries.values_at_bounds import ValuesAtBounds

# %% [markdown]
# ## Set up pint

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %% [markdown]
# ## The `TimeseriesContinuous` class

# %%
time_axis = Q([2020, 2030, 2050], "yr")
ms = Q([1.0, 2.0], "Gt / yr / yr")
cs = Q([10.0, 20.0], "Gt / yr")

# %%
piecewise_polynomial_constant = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.atleast_2d(cs.m),
)

# %%
# notes for docs
# - never initialise like this, almost always have your own thing or use `from_discrete` or similar
#   (add examples below)
# - interface for function is defined in ...
# - under the hood, this just uses interpolate, which is a trivial operation on a continuous function
# - add extrapolation examples
ts = TimeseriesContinuous(
    name="piecewise_constant",
    time_units=time_axis.u,
    values_units=cs.u,
    function=piecewise_polynomial_constant,
)
ts

# %%
time_axis_plot = TimeAxis(time_axis)

# %%
UR.setup_matplotlib(enable=True)

# %%
ts.plot(time_axis_plot)

# %%
fig, ax = plt.subplots()

for res_increase in (1, 5, 10, 100, 300):
    ts.plot(
        time_axis_plot,
        ax=ax,
        res_increase=res_increase,
        label=f"{res_increase=}",
        alpha=0.7,
        linestyle="--",
    )

ax.legend()

# %%
piecewise_polynomial_linear = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.vstack(
        [
            ms.m,
            cs.m,
        ]
    ),
)

ts_linear = TimeseriesContinuous(
    name="piecewise_linear",
    time_units=time_axis.u,
    values_units=cs.u,
    function=piecewise_polynomial_linear,
)

# %%
fig, ax = plt.subplots()

for res_increase in (1, 5, 10, 100, 300):
    ts_linear.plot(
        time_axis_plot,
        ax=ax,
        res_increase=res_increase,
        label=f"{res_increase=}",
        alpha=0.7,
        linestyle="--",
    )

ax.legend()

# %%
a_values = Q([0.1, -0.05], "Gt / yr / yr / yr")
b_values = Q([0.0, 2.0], "Gt / yr / yr")
c_values = cs

# %%
piecewise_polynomial_quadratic = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.vstack(
        [
            a_values.m,
            b_values.m,
            c_values.m,
        ]
    ),
)

ts_quadratic = TimeseriesContinuous(
    name="piecewise_quadratic",
    time_units=time_axis.u,
    values_units=cs.u,
    function=piecewise_polynomial_quadratic,
)

# %%
fig, ax = plt.subplots()

for res_increase in (1, 2, 5, 100):
    ts_quadratic.plot(
        time_axis_plot,
        ax=ax,
        res_increase=res_increase,
        label=f"{res_increase=}",
        alpha=0.7,
        linestyle="--",
    )

ax.legend()

# %%
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))

for ts_plot in (ts, ts_linear, ts_quadratic):
    ts_plot.plot(time_axis_plot, ax=axes[0], alpha=0.7, linestyle="--")
    ts_plot.differentiate().plot(time_axis_plot, ax=axes[1], alpha=0.7, linestyle="--")

    integration_constant = Q(0, "Gt")
    # Need to fix up interfaces/wrapper class for this
    # ts_plot.integrate(integration_constant=integration_constant).plot(time_axis_plot, ax=axes[2], alpha=0.7, linestyle="--")

    integral_values_units = ts_plot.values_units * ts_plot.time_units

    tmp = ts_plot.function.antiderivative()
    c_new = tmp.c
    c_new[-1, :] += integration_constant.to(integral_values_units).m

    # TODO: introduce wrapper class to help clean this interface up
    # to make writing the Protocol easier.
    function_integral = scipy.interpolate.PPoly(
        c=c_new,
        x=tmp.x,
        extrapolate=False,
    )
    ts_integral = TimeseriesContinuous(
        name=f"{ts_plot.name}_integral",
        time_units=ts_plot.time_units,
        values_units=ts_plot.values_units,
        function=function_integral,
    )

    ts_integral.plot(time_axis_plot, ax=axes[2], alpha=0.7, linestyle="--")


for ax in axes:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

fig.tight_layout()

# %%
ts_quadratic.interpolate(time_axis_plot)

# %%
ts_quadratic.interpolate(Q([2010, 2020, 2030, 2050, 2060], "yr"))

# %%
ts_quadratic.interpolate(
    Q([2010, 2020, 2030, 2050, 2060], "yr"), allow_extrapolation=True
)

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
