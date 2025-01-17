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

# %%
import multiprocessing

import matplotlib.pyplot as plt
import pandas as pd
import pint

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.timeseries_continuous import get_plot_points
import continuous_timeseries.pandas_accessors

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %%
UR.setup_matplotlib(enable=True)

# %%
continuous_timeseries.pandas_accessors.register_pandas_accessor()

# %%
x = Q([2010, 2015, 2025], "yr")
y_ms = [
    [100.0, 200.0, 350.0],
    [-1.5, -0.5, 0.5],
]
idx = pd.MultiIndex.from_tuples(
    (
        ("name_1", "World", "Mt  /yr"),
        ("name_2", "World", "Gt / yr"),
    ),
    # units not unit to follow pint conventions
    names=["name", "region", "units"],
)

df = pd.DataFrame(
    y_ms,
    columns=x.m,
    index=idx,
)
df

# %%
series = df.ct.to_timeseries_two(
    time_units=x.units,
    # interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed
    interpolation=InterpolationOption.Linear
)
series

# %%
df.ct.to_timeseries_two(
    time_units=x.units,
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    progress=True,
)

# %%
df.ct.to_timeseries_two(
    time_units=x.units,
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    progress=True,
    n_processes=2,
    mp_context=multiprocessing.get_context("fork"),
)

# %%
series = df.ct.to_timeseries(
    time_units=x.units,
    # interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed
    interpolation=InterpolationOption.Linear
)
series

# %%
series.ct.metadata

# %%
plot_points = get_plot_points(series.iloc[0].time_axis.bounds, res_increase=100)

# %%
series.ct.interpolate_two(plot_points, progress=False, n_processes=1)

# %%
series.ct.interpolate_two(plot_points, progress=True, n_processes=1)

# %%
series.ct.interpolate_two(plot_points, progress=True, n_processes=2)

# %%
series.ct.interpolate_two(
    plot_points, 
    progress=True, 
    n_processes=2, 
    mp_context=multiprocessing.get_context("fork"),
)

# %%
# Would want to be able to do this in parallel too.
derivative = series.map(lambda x: x.differentiate())
integral = series.map(lambda x: x.integrate(Q(0.0, "Gt")))

# %%
# API to aim for:
# series.ct.differentiate().ct.plot()
# series.loc[pix.isin(variable="Emissions|CO2")].ct.differentiate().ct.plot()

# %%
# Want basic label control for plotting.
# For everything else, drop out to seaborn.
fig, ax = plt.subplots()
for idx, row in integral.items():
    row.plot(ax=ax, continuous_plot_kwargs=dict(label=idx))
    row.plot(ax=ax)

ax.legend()

# %%
# hyperfine plotting control
# filter to something that will all have same units first (use pandas_indexing)
# then interpolate (maybe need to put an `increase_resolution` method on Timeseries)
# then get dataframe with uniform units
# then melt (or maybe just make a 'get_sns_df' accessor that includes a check of units)
# then plot using seaborn

# %%
plot_points = get_plot_points(series.iloc[0].time_axis.bounds, res_increase=100)
plot_points

# %%
fine = series.map(lambda x: x.interpolate(plot_points))
fine = integral.map(lambda x: x.interpolate(plot_points))

# %%
import numpy as np
time_units = "yr"
out_units = "Mt"

tmp_l = []
for i, (k, v) in enumerate(fine.items()):
    # .to_series() for individual Timeseries
    discrete = v.discrete
    columns = discrete.time_axis.bounds.to(time_units).m
    values = discrete.values_at_bounds.values
    if out_units:
        values = values.to(out_units)
        units = out_units
    else:
        units = str(values.u)
    
    tmp = pd.Series(
        values.m,
        index=columns,
    )
    # end of .to_series() for individual Timeseries

    # Join on metadata and convert to DataFrame
    index = pd.MultiIndex.from_tuples(
        ((*k, units),),
        names=[*fine.index.names, "units"]
    )
    tmp = pd.DataFrame(tmp, columns=index).T

    tmp_l.append(tmp)

# Create the result (can use pd.concat
# as we're guaranteed to have the index in the right order)
res = pd.concat(tmp_l)
res

# %%
import seaborn as sns

# %%
sns_df = res.melt(
    var_name="time",
    ignore_index=False,
).reset_index()
sns_df

# %%
sns.lineplot(
    data=sns_df,
    x="time",
    y="value",
    hue="name",
    style="region",
)

# %%
n_variables = 3
n_yrs = 250
n_runs = 10
n_scenarios = 5

n_variables = 1
n_yrs = 550
n_runs = 600
n_scenarios = 100

# n_variables = 1
# n_yrs = 125
# n_runs = 600
# n_scenarios = 1000

# # Too big, not really possible to do in memory
# n_variables = 10
# n_yrs = 550
# n_runs = 600
# n_scenarios = 2000

# %%
x = Q(np.arange(n_yrs) + 1750, "yr")
x

# %%
y_ms = np.random.random((n_variables * n_runs * n_scenarios, n_yrs))
y_ms.shape

# %%
import itertools

# %%
idx = pd.MultiIndex.from_tuples(
    ((s, v, r, "Mt / yr") for s, v, r in itertools.product(
    [f"variable_{i}" for i in range(n_variables)],
    [f"scenario_{i}" for i in range(n_scenarios)],
    [i for i in range(n_runs)],
)
), names=["scenario", "variable", "region", "units"])
idx

# %%
df = pd.DataFrame(
    y_ms,
    columns=x.m,
    index=idx,
)
df

# %%
# TODO: drop nans when converting
# TODO: test with a dataframe that has history and scenario, but no overlap

# %%
series_h = df.ct.to_timeseries_two(
    time_units=x.units,
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    progress=True,
    n_processes=1, 
)

# %%
series_h = df.ct.to_timeseries_two(
    time_units=x.units,
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    progress=True,
    n_processes=multiprocessing.cpu_count(), 
    mp_context=multiprocessing.get_context("fork"),
)

# %%
# Must be something smarter that can be done with chunking to make this faster, anyway
series_h = df.ct.to_timeseries(
    time_units=x.units, 
    interpolation=InterpolationOption.Linear, 
    progress=True, 
    n_processes=multiprocessing.cpu_count(), 
    mp_context=multiprocessing.get_context("fork"),
)

# %%
series_h.ct.integrate(
    Q(0, "Mt"),
    progress=True,
    n_processes=1,
)

# %%
# series_h.ct.integrate(
#     Q(0, "Mt"), 
#     progress=True, 
#     n_processes=multiprocessing.cpu_count(),
# )

# %%
series_h.ct.integrate(
    Q(0, "Mt"), 
    progress=True, 
    # n_processes=multiprocessing.cpu_count(),
    n_processes=3,
    mp_context=multiprocessing.get_context("fork"),
)

# %%
interp_points = get_plot_points(series_h.iloc[0].time_axis.bounds, res_increase=100)

# %%
series_h.ct.interpolate_two(interp_points, progress=True, n_processes=1)

# %%
# series_h.ct.interpolate_two(
#     interp_points, 
#     progress=True, 
#     n_processes=multiprocessing.cpu_count(),
# )

# %%
# %%time
series_h.ct.interpolate_two(
    interp_points, 
    progress=False, 
    n_processes=1,
)

# %%
# %%time
series_h.ct.interpolate_two(
    interp_points, 
    progress=False, 
    n_processes=multiprocessing.cpu_count() + 4,
    mp_context=multiprocessing.get_context("fork"),
)

# %%
series_h.ct.interpolate_two(
    interp_points, 
    progress=True, 
    n_processes=multiprocessing.cpu_count(),
    mp_context=multiprocessing.get_context("fork"),
)

# %%
# # It could also be something to do with pint that makes the parallel processing slow...
# # Some links to think about re parallelisation:
# # - https://medium.com/@codewithnazam/pandas-in-a-parallel-universe-speeding-up-your-data-adventures-7696aa00eab8
# # - https://pypi.org/project/parallel-pandas/
# # - https://towardsdatascience.com/easily-parallelize-your-calculations-in-pandas-with-parallel-pandas-dc194b82d82f
# # - https://github.com/nalepae/pandarallel
# series_h.ct.interpolate(interp_points, progress=True, n_processes=multiprocessing.cpu_count())

# %%
series_h.ct.interpolate(interp_points, progress=True, n_processes=1)

# %%
series_h.ct.interpolate(interp_points, progress=False, n_processes=multiprocessing.cpu_count())

# %%
series_h.ct.interpolate(interp_points, progress=False)

# %%
# df.ct.to_timeseries(time_units=x.units, interpolation=InterpolationOption.Linear, progress=True, n_processes=multiprocessing.cpu_count(), mp_context=multiprocessing.get_context("spawn"))

# %%
df.ct.to_timeseries(time_units=x.units, interpolation=InterpolationOption.Linear, progress=True, n_processes=1)

# %%
df.ct.to_timeseries(time_units=x.units, interpolation=InterpolationOption.Linear, n_processes=multiprocessing.cpu_count())

# %%
df.ct.to_timeseries(time_units=x.units, interpolation=InterpolationOption.Linear, n_processes=1)
