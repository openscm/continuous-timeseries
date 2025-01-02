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
import numpy as np
import pint
import scipy.interpolate

from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
    TimeseriesContinuous,
)

# %% [markdown]
# ## Set up pint
#
# For details, see the pint docs
# ([stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html),
# [last version that we checked at the time of writing](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))
# [or our docs on unit-aware plotting](../discrete_timeseries_tutorial#unit-aware-plotting).  # noqa: E501

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %% [markdown]
# ## Set up matplotlib to work with pint

# %%
UR.setup_matplotlib(enable=True)

# %% [markdown]
# ## The `TimeseriesContinuous` class
#
# The `TimeseriesContinuous` is our representation of continuous timeseries.
# It needs a few pieces.
# The first is a name, this is straight-forward.
# The next pieces are the units of time (`time_units`),
# the units for the values in the timeseries (`values_units`)
# and a class which holds
# the continuous representation of the timeseries.
# This class must match the interface defined by
# `continuous_timeseries.timeseries_continuous.ContinuousFunctionLike`,
# i.e. it should support evaluating the function,
# integration and differentiation.
#
# We also provide a concrete implementation of a class that satisfies the
# `continuous_timeseries.timeseries_continuous.ContinuousFunctionLike`
# interface. It is `ContinuousFunctionScipyPPoly`.
# This is a thin wrapper around scipy's
# `scipy.interpolate.PPoly` class.
# We use the `ContinuousFunctionScipyPPoly` class throughout these docs,
# but things are written such that other representations of continuous timeseries
# could be used if desired
# (for example, many of
# [scipy's other polynomial representations](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
# ).

# %% [markdown]
# ### Setting up our continuous representation

# %% [markdown]
# #### Piecewise constant
#
# Here we create an instance of `ContinuousFunctionScipyPPoly`
# that represents data that is stepwise constant i.e. its value
# is constant over each timestep.

# %%
time_axis = Q([2020, 2030, 2050], "yr")

values = Q([10.0, 20.0], "Gt / yr")

piecewise_polynomial_constant = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.atleast_2d(values.m),
)
continuous_constant = ContinuousFunctionScipyPPoly(piecewise_polynomial_constant)
continuous_constant

# %% [markdown]
# We then create our `TimeseriesContinuous` instance.

# %%
ts = TimeseriesContinuous(
    name="piecewise_constant",
    time_units=time_axis.u,
    values_units=values.u,
    function=continuous_constant,
)
ts

# %% [markdown]
# If we plot this data, it is much clearer what is going on.
# (As a note, when plotting,
# we need to specify the time axis we want to use for plotting
# (continuous representations don't carry this information around with them)).

# %%
ts.plot(time_axis=time_axis)

# %% [markdown]
# #### Linear
#
# Here we create an instance of `ContinuousFunctionScipyPPoly`
# that represents data that is linear between the defined discrete values.

# %%
gradients = Q([1.0, 1.5], values.units / time_axis.units)

piecewise_polynomial_linear = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.vstack([gradients.m, values.m]),
)
continuous_linear = ContinuousFunctionScipyPPoly(piecewise_polynomial_linear)

ts_linear = TimeseriesContinuous(
    name="piecewise_linear",
    time_units=time_axis.u,
    values_units=values.u,
    function=continuous_linear,
)
ts_linear

# %%
ts_linear.plot(time_axis=time_axis)

# %% [markdown]
# #### Quadratic
#
# We also create an instance of `ContinuousFunctionScipyPPoly`
# that represents data that is quadratic between the defined discrete values.

# %%
a_values = Q([0.1, -1 / 40], values.u / time_axis.u / time_axis.u)
b_values = Q([0.0, 2.0], values.u / time_axis.u)
c_values = values

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
continuous_quadratic = ContinuousFunctionScipyPPoly(piecewise_polynomial_quadratic)

ts_quadratic = TimeseriesContinuous(
    name="piecewise_quadratic",
    time_units=time_axis.u,
    values_units=values.u,
    function=continuous_quadratic,
)
ts_quadratic

# %%
ts_quadratic.plot(time_axis=time_axis)

# %% [markdown]
# #### Cubic
#
# We also create an instance of `ContinuousFunctionScipyPPoly`
# that represents data that is cubic between the defined discrete values.

# %%
a_values = Q([-1 / 100, 5 / 800], values.u / time_axis.u**3)
b_values = Q([0.2, -0.1], values.u / time_axis.u**2)
c_values = Q([0.0, 1.0], values.u / time_axis.u)
d_values = values

piecewise_polynomial_quadratic = scipy.interpolate.PPoly(
    x=time_axis.m,
    c=np.vstack(
        [
            a_values.m,
            b_values.m,
            c_values.m,
            d_values.m,
        ]
    ),
)
continuous_quadratic = ContinuousFunctionScipyPPoly(piecewise_polynomial_quadratic)

ts_cubic = TimeseriesContinuous(
    name="piecewise_cubic",
    time_units=time_axis.u,
    values_units=values.u,
    function=continuous_quadratic,
)
ts_cubic

# %%
ts_cubic.plot(time_axis=time_axis)

# %% [markdown]
# ### Comparing the interpolation choices
#
# If we plot the different interpolation choices on the same axes,
# the difference between them is clear.
# This also makes clear why having continuous representations is helpful:
# they add information that cannot be captured by discrete representations alone
# (all the timeseries go through the same discrete points,
# but are completely different in between).

# %%
fig, ax = plt.subplots()

for ts_plot in (ts, ts_linear, ts_quadratic, ts_cubic):
    ts_plot.plot(time_axis, ax=ax, alpha=0.7, linestyle="--")

ax.set_ylim(ymin=0)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

fig.tight_layout()

# %% [markdown]
# ### Intepolation
#
# With our continuous representations,
# interpolation is trivial.

# %%
ts_quadratic.interpolate(Q([2023, 2025, 2030, 2035], "yr"))

# %%
ts_quadratic.interpolate(Q([2020, 2025, 2045], "yr"))

# %% [markdown]
# Extrapolation is also possible.

# %%
ts_quadratic.interpolate(Q([2000, 2025, 2055], "yr"), allow_extrapolation=True)

# %% [markdown]
# However, if you don't explicitly allow it,
# you will get an error if you try and extrapolate.

# %%
try:
    ts_quadratic.interpolate(Q([2000, 2025, 2055], "yr"))
except ValueError:
    traceback.print_exc(limit=0)

# %% [markdown]
# ### Integration and differentiation
#
# With our continuous representations,
# integration and differentiation are also trivial.

# %%
fig, axes_ar = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes_ar.flatten()

for ts_plot in (ts, ts_linear, ts_quadratic, ts_cubic):
    ts_plot.plot(time_axis, ax=axes[0], alpha=0.7, linestyle="--")
    ts_plot.differentiate().plot(time_axis, ax=axes[1], alpha=0.7, linestyle="--")

    integration_constant = Q(0, "Gt")
    integral = ts_plot.integrate(integration_constant=integration_constant)
    integral.plot(time_axis, ax=axes[2], alpha=0.7, linestyle="--")

    integral.integrate(integration_constant=Q(0.0, "Gt yr")).plot(
        time_axis, ax=axes[3], alpha=0.7, linestyle="--"
    )

axes[0].set_ylim(ymin=0)
for ax in axes:
    ax.legend()

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
fig, axes = plt.subplots(nrows=4, figsize=(12, 12))

for i, ts_plot in enumerate((ts, ts_linear, ts_quadratic, ts_cubic)):
    for res_increase in (1, 5, 100, 300):
        ts_plot.plot(
            time_axis,
            ax=axes[i],
            res_increase=res_increase,
            label=f"{res_increase=}",
            alpha=0.7,
            linestyle="--",
        )

    axes[i].set_title(ts_plot.name)
    axes[i].set_ylim(ymin=0)
    axes[i].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

fig.tight_layout()

# %% [markdown]
# #### Customising plots and other features

# %% [markdown]
# By default, the plotted points
# are labelled with the name of the `TimeseriesContinuous` object.
# This is shown if you add a legend to your plot.

# %%
ax = ts_linear.plot(time_axis=time_axis)
ax.legend()

# %% [markdown]
# The `label` argument and any unrecognised arguments
# are simply passed through to the `plot` method of the underlying axes.
# This gives you full control over the plot.

# %%
fig, ax = plt.subplots()

y_unit = "Mt / year"
ax.set_ylabel(y_unit)
ax.yaxis.set_units(UR.Unit(y_unit))

ts_linear.plot(
    time_axis=time_axis,
    ax=ax,
    res_increase=2,
    marker="x",
    color="tab:orange",
    label="demo",
    linestyle="--",
    linewidth=2,
)
ts_quadratic.plot(
    time_axis=time_axis,
    ax=ax,
    res_increase=2,
    marker="o",
    color="tab:cyan",
    label="second demo",
    linestyle=":",
)

ax.set_ylim(0.0)
ax.grid()
ax.legend()
