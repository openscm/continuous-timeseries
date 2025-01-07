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
# # Budget-compatible pathways
#
# Here we discuss our module
# for creating emissions pathways compatible with a given budget.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint
import scipy.interpolate
import scipy.optimize

import continuous_timeseries as ct
import continuous_timeseries.budget_compatible_pathways as ct_bcp
from continuous_timeseries.timeseries_continuous import ContinuousFunctionScipyPPoly

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
# ## Calculating a budget-compatible pathway

# %% [markdown]
# Let's imagine we start with some emissions budget from some point in time.

# %%
budget = Q(500, "GtCO2")
budget_start_time = Q(2020.0, "yr")

# %% [markdown]
# ### Linear pathway to zero
#
# Calculating a linear pathway to zero emissions
# that is compatible with this budget is trivial.
# The only other piece of information you need
# is the emissions you want to start from.
# Normally, that will be emissions at the time
# from which the budget is/was available.

# %%
emissions_start = Q(10.4, "GtC / yr").to("GtCO2 / yr")

# %%
linear_in_line_with_budget = ct_bcp.derive_linear_path(
    budget=budget,
    budget_start_time=budget_start_time,
    emissions_start=emissions_start,
)
linear_in_line_with_budget

# %%
fig, axes = plt.subplots(nrows=2, sharex=True)

linear_in_line_with_budget.plot(ax=axes[0])
linear_in_line_with_budget.integrate(Q(0, "GtC"), name_res="Cumulative emissions").plot(
    ax=axes[1]
)

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()

# %% [markdown]
# There is one thing to notice on this plot.
# The pathway is extended slightly beyond the net-zero year with zero emissions.
# This ensures that there is at least one full year with zero emissions
# and you get sensible results if you extrapolate after creating the pathway.

# %%
extrapolation_times = np.arange(
    np.ceil(budget_start_time).to("yr").m, 2100.0 + 1.0, 1.0
) * UR.Unit("yr")

linear_extrapolated = linear_in_line_with_budget.interpolate(
    extrapolation_times, allow_extrapolation=True
)
linear_extrapolated.timeseries_continuous.name = "Extrapolated"

fig, axes = plt.subplots(nrows=2, sharex=True)


linear_extrapolated.plot(ax=axes[0])
linear_extrapolated.integrate(
    Q(0, "GtC"), name_res="Extrapolated cumulative emissions"
).plot(ax=axes[1])

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()

# %% [markdown]
# #### Annual steps
#
# In general, countries don't report emissions instantaneously.
# Instead, they report emissions as the average over each year.
# With Continuous Timeseries, it is trivial to transform our pathway
# into something which can be directly compared with country emissions.

# %%
linear_annual_steps_in_line_with_budget = ct_bcp.convert_to_annual_constant_emissions(
    linear_in_line_with_budget, name_res="Annualised"
)

# %%
continuous_plot_kwargs = dict(linewidth=2.0, alpha=0.7)
pathways = [linear_in_line_with_budget, linear_annual_steps_in_line_with_budget]

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

for pw in pathways:
    pw.plot(ax=axes[0], continuous_plot_kwargs=continuous_plot_kwargs)
    pw.integrate(Q(0, "GtC"), name_res=f"Cumulative emissions {pw.name}").plot(
        ax=axes[1], continuous_plot_kwargs=continuous_plot_kwargs
    )

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()

# %% [markdown]
# The difference between the two clearer if we take a more extreme example.

# %%
budget_small = Q(0.425, "GtCO2")
budget_small_start_time = Q(2020.0, "yr")
emissions_small_start = Q(300, "MtCO2 / yr")

# %%
linear_in_line_with_budget_small = ct_bcp.derive_linear_path(
    budget=budget_small,
    budget_start_time=budget_small_start_time,
    emissions_start=emissions_small_start,
)
linear_annual_steps_in_line_with_budget_small = (
    ct_bcp.convert_to_annual_constant_emissions(
        linear_in_line_with_budget_small, name_res="Annualised"
    )
)

# %%
continuous_plot_kwargs = dict(linewidth=2.0, alpha=0.7)
pathways = [
    linear_in_line_with_budget_small,
    linear_annual_steps_in_line_with_budget_small,
]

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

for pw in pathways:
    pw.plot(ax=axes[0], continuous_plot_kwargs=continuous_plot_kwargs)
    pw.integrate(Q(0, "GtC"), name_res=f"Cumulative emissions {pw.name}").plot(
        ax=axes[1], continuous_plot_kwargs=continuous_plot_kwargs
    )

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()

# %% [markdown]
# This example also shows why annualising emissions can be tricky.
# The drop from one year to the next is the same for all steps,
# except the year in which (instantaneous) net zero is reached,
# which has a smaller drop to the zero emissions.
# For example, in the annualised pathway above,
# the drop in emissions from 2021 to 2022 is around 100 MtCO<sub>2</sub> / yr
# while the drop in emissions from 2022 to 2023 is only around 40 MtCO<sub>2</sub> / yr.
# (There are ways to do such calculations in their discrete form,
# Continuous Timeseries just removes those headaches.)

# %% [markdown]
# ### Curved pathway to zero
#
# A linear pathway is one way to get to zero.
# Other options are possible, for example a smoother pathway.
#
# #### Quadratic
#
# First we show a symmetric quadratic.
# This generally produces nice results.
# Its major downside is that its gradient is zero today,
# which may not be desirable in all cases.

# %%
quadratic_in_line_with_budget = ct_bcp.derive_symmetric_quadratic_path(
    budget=budget,
    budget_start_time=budget_start_time,
    emissions_start=emissions_start,
)

# %%
continuous_plot_kwargs = dict(linewidth=2.0, alpha=0.7)
pathways = [linear_in_line_with_budget, quadratic_in_line_with_budget]

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

for pw in pathways:
    pw.plot(ax=axes[0], continuous_plot_kwargs=continuous_plot_kwargs)
    pw.integrate(Q(0, "GtC"), name_res=f"Cumulative emissions {pw.name}").plot(
        ax=axes[1], continuous_plot_kwargs=continuous_plot_kwargs
    )

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()

# %%
quadratic_annual_steps_in_line_with_budget = (
    ct_bcp.convert_to_annual_constant_emissions(
        quadratic_in_line_with_budget, name_res="Annualised quadratic"
    )
)

# %%
continuous_plot_kwargs = dict(linewidth=2.0, alpha=0.7)
pathways = [quadratic_in_line_with_budget, quadratic_annual_steps_in_line_with_budget]

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

for pw in pathways:
    pw.plot(ax=axes[0], continuous_plot_kwargs=continuous_plot_kwargs)
    pw.integrate(Q(0, "GtC"), name_res=f"Cumulative emissions {pw.name}").plot(
        ax=axes[1], continuous_plot_kwargs=continuous_plot_kwargs
    )

for ax in axes:
    ax.legend()
    ax.grid()

fig.tight_layout()


# %% [markdown]
# #### Custom
#
# Here we show how to make a custom fit,
# in case you want to explore further yourself.
# Here we do the fit such that the emissions and gradient today
# are specified, but we still stay within the budget.
# We do this by fitting a cubic pathway.


# %%
def get_cubic_pathway(  # noqa: D103, PLR0913
    budget,
    budget_start_time,
    emissions_start,
    emissions_gradient_start,
    name_res,
    max_iter: int = 100,
):
    time_units = budget_start_time.u
    values_units = emissions_start.u

    E_0 = emissions_start
    m_0 = emissions_gradient_start

    # Use linear net-zero as our initial guess
    linear_nz_year = ct_bcp.calculate_linear_net_zero_time(
        budget,
        budget_start_time,
        emissions_start,
    )
    # Non-linear equations, beyond my pay grade.
    # Time to bring out the big guns
    budget_start_time_m = budget_start_time.to(time_units).m
    budget_m = budget.m

    m_0_m = m_0.to(values_units / time_units).m
    E_0_m = E_0.to(values_units).m

    def get_a(nzd, m_0, E_0):
        return (m_0 * nzd + 2 * E_0) / (nzd**3)

    def get_b(nzd, m_0, E_0):
        return -(2 * m_0 * nzd + 3 * E_0) / (nzd**2)

    def get_budget_diff(x):
        nzd = x - budget_start_time_m
        time_bounds_m = np.hstack([budget_start_time_m, x])

        a_m = get_a(nzd=nzd, m_0=m_0_m, E_0=E_0_m)
        b_m = get_b(nzd=nzd, m_0=m_0_m, E_0=E_0_m)

        c_non_zero_m = np.array([[a_m], [b_m], [m_0_m], [E_0_m]])

        ppoly_raw = scipy.interpolate.PPoly(x=time_bounds_m, c=c_non_zero_m)
        budget_raw = ppoly_raw.integrate(budget_start_time_m, x)

        budget_diff = budget_raw - budget_m

        return budget_diff

    nz_yr_m, res_scipy = scipy.optimize.newton(
        func=get_budget_diff,
        x0=linear_nz_year.m,
        full_output=True,
        maxiter=500,
    )
    if not res_scipy.converged:
        raise AssertionError

    nz_yr = nz_yr_m * linear_nz_year.u

    nzd = nz_yr - budget_start_time
    a = get_a(nzd=nzd, m_0=m_0, E_0=E_0)
    b = get_b(nzd=nzd, m_0=m_0, E_0=E_0)

    c_non_zero = np.array(
        [
            [a.to(values_units / time_units**3).m],
            [b.to(values_units / time_units**2).m],
            [m_0.to(values_units / time_units).m],
            [E_0.to(values_units).m],
        ]
    )

    # Add on the zero component
    c = np.hstack([c_non_zero, np.zeros((c_non_zero.shape[0], 1))])

    last_ts_time = np.floor(nz_yr) + 2.0 * nz_yr.to("yr").u
    x_bounds = np.hstack([budget_start_time, nz_yr, last_ts_time])

    x = x_bounds.to(time_units).m
    ppoly = scipy.interpolate.PPoly(x=x, c=c)

    tsc = ct.TimeseriesContinuous(
        name=name_res,
        time_units=time_units,
        values_units=values_units,
        function=ContinuousFunctionScipyPPoly(ppoly),
        domain=(budget_start_time, last_ts_time),
    )
    ts = ct.Timeseries(
        time_axis=ct.TimeAxis(x_bounds),
        timeseries_continuous=tsc,
    )

    return ts


# %%
emissions_gradient_start = Q(-1.0, "GtCO2 / yr / yr")
times_plot = (
    np.arange(budget_start_time.to("yr").m, 2081, 1.0) * budget_start_time.to("yr").u
)

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
for emissions_gradient_start in Q([-2.0, 0.0, 5.0, 7.5], "GtCO2 / yr / yr"):
    cubic = get_cubic_pathway(
        budget=budget,
        budget_start_time=budget_start_time,
        emissions_start=emissions_start,
        emissions_gradient_start=emissions_gradient_start,
        name_res=f"Initial gradient: {emissions_gradient_start:.2f}",
    ).interpolate(times_plot, allow_extrapolation=True)

    annualised = ct_bcp.convert_to_annual_constant_emissions(
        cubic, name_res=f"Annualised {cubic.name}"
    )

    cubic.plot(ax=axes[0], continuous_plot_kwargs=dict(linewidth=2.0))
    annualised.plot(
        ax=axes[0], continuous_plot_kwargs=dict(label="", zorder=1.0, alpha=0.6)
    )
    cubic.differentiate().plot(ax=axes[1])
    cubic.integrate(Q(0, "GtC")).plot(ax=axes[2])

axes[0].legend()
axes[2].legend()
for ax in axes:
    ax.grid()

fig.tight_layout()
