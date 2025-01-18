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
# # Pandas accessors API
#
# Here we introduce our Pandas accessor API.
# There make use of
# [pandas accessors API](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors)
# to create an easy integration between Continuous Timeseries and pandas objects.
#
# For many use cases, simply using pandas objects directly is the best option.
# However, the interoperability makes it easy to convert between,
# so you get the best of both worlds.

# %% [markdown]
# ## Imports

# %%
import itertools
import multiprocessing
import traceback

import numpy as np
import openscm_units
import pandas as pd
import pandas_indexing as pix
import pint
import seaborn as sns

import continuous_timeseries as ct
import continuous_timeseries.pandas_accessors

# %% [markdown]
# ## Registering the accessors
#
# You must register the accessors before they can be used.
# We make this step explicit so that our imports don't have side effects
# (we've had bad experiences with imports with side effects
# and don't want you to have to those bad experiences too).

# %%
# If you try and use the accessor before registering,
# you will get an AttributeError.
try:
    pd.Series.ct
except AttributeError:
    traceback.print_exc(limit=0)

# %%
continuous_timeseries.pandas_accessors.register_pandas_accessor()

# %%
# Having registered the accessor,
# the "ct" namespace is now available.
pd.Series.ct

# %% [markdown]
# ## Set up pint

# %%
pint.set_application_registry(openscm_units.unit_registry)

# %% [markdown]
# ## Handy pint aliases

# %%
UR = pint.get_application_registry()

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
# ## Helper functions


# %%
def create_df(
    *,
    n_scenarios: int,
    n_variables: int,
    n_runs: int,
    timepoints: np.typing.NDArray[np.floating],
    units: str = "Mt / yr",
) -> pd.DataFrame:
    """
    Create an example `pd.DataFrame`

    This uses the idea of simple climate model runs,
    where you have a number of scenarios,
    each of which has a number of variables
    from a number of different model runs
    with output for a number of different time points.
    """
    idx = pd.MultiIndex.from_frame(
        pd.DataFrame(
            (
                (s, v, r, units)
                for s, v, r in itertools.product(
                    [f"scenario_{i}" for i in range(n_scenarios)],
                    [f"variable_{i}" for i in range(n_variables)],
                    [i for i in range(n_runs)],
                )
            ),
            columns=["scenario", "variable", "run", "units"],
            # This makes updates and general handling later way way faster.
            # TODO: make this tip clearer.
            dtype="category",
        )
    )

    n_ts = n_scenarios * n_variables * n_runs
    df = pd.DataFrame(
        50.0 * np.linspace(0.3, 1, n_ts)[:,  np.newaxis] * np.linspace(0, 1, timepoints.size)[np.newaxis, :] + np.random.random((n_ts, timepoints.size)),
        columns=timepoints,
        index=idx,
    )

    return df


# %% [markdown]
# ## Converting to `Timeseries`
#
# Here we show how to convert to `Timeseries`.
# More specifically, how to take a `pd.DataFrame`
# and convert it to a `pd.Series` of `Timeseries`.
# The benefit here is that you can still filter/manipulate the result
# using standard pandas filtering,
# but you have `Timeseries` objects to work with from there.

# %% [markdown]
# We start with a basic `pd.DataFrame`.

# %%
small_df = create_df(
    n_scenarios=3,
    n_variables=2,
    n_runs=5,
    timepoints=np.arange(250) + 1850.0,
)
small_df

# %% [markdown]
# Then we convert it time series.

# %%
small_ts = small_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
)
small_ts

# %% [markdown]
# Then we can use standard Continuous timeseries APIs,
# e.g. plotting.

# %%
small_ts.ct.plot(continuous_plot_kwargs=dict(alpha=0.3))
# # TODO: move this to plotting
# small_ts.ct.plot(continuous_plot_kwargs=dict(alpha=0.3), progress=True)

# %% [markdown]
# When combined with [pandas-indexing](https://pandas-indexing.readthedocs.io/en/latest/index.html),
# this can be quite powerful for quick plots.

# %%
ax = small_ts.loc[pix.isin(variable="variable_0")].ct.plot(continuous_plot_kwargs=dict(alpha=0.3))
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))

# %%
# TODO: move this to plotting section
ax = small_ts.loc[pix.isin(variable="variable_0", run=0)].ct.plot(label="scenario", continuous_plot_kwargs=dict(alpha=0.9))
ax.legend()

# %% [markdown]
# If we have a bigger `pd.DataFrame`, the conversion process can be much slower.

# %%
bigger_df = create_df(
    n_scenarios=100,
    n_variables=2,
    n_runs=300,
    timepoints=np.arange(351) + 1850.0,
)
bigger_df.shape

# %% [markdown]
# If want to see the conversion's progress,
# you can activate the progress bar if you have
# [`tdqm`](https://tqdm.github.io/) installed.

# %%
bigger_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.Linear,
    progress=True,
)

# %% [markdown]
# If you want to speed things up,
# you may want to process the `pd.DataFrame` in parallel.

# %%
n_processes = multiprocessing.cpu_count()
n_processes

# %%
bigger_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.Linear,
    n_processes=n_processes,
)

# %% [markdown]
# If you want progress bars in parallel,
# we support that too.

# %%
bigger_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.Linear,
    n_processes=n_processes,
    progress=True,
)

# %% [markdown]
# If you want nested progress bars in parallel,
# we support that too
# (although we're not sure if this works on windows
# because of the need for forking, for details see
# [here](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)).

# %%
bigger_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.Linear,
    n_processes=n_processes,
    progress=True,
    # We have found that nested progress bars in parallel
    # only really work if we use forking
    # (on windows, probably best to just not use the nested bars
    # because forking isn't supported).
    progress_nested=True,
    mp_context=multiprocessing.get_context("fork"),
)

# %% [markdown]
# On big `pd.DataFrame`'s the combination with
# [pandas indexing](https://pandas-indexing.readthedocs.io/)
# becomes particularly powerful.

# %%
ax = (
    bigger_df
    .loc[pix.isin(variable="variable_1")]
    .groupby(["scenario", "variable", "units"], observed=True)
    .median()
    .loc[pix.ismatch(scenario="scenario_1*")]
    .ct.to_timeseries(
        time_units="yr",
        interpolation=ct.InterpolationOption.Quadratic,
    )
    .ct.plot()
)
ax.legend()

# %%
# # Units don't round trip
# pd.testing.assert_frame_equal(
#     small_df,
#     small_ts.ct.to_df()
# )
small_ts.ct.to_df()

# %%
small_ts.ct.to_df(increase_resolution=3)

# %%
sns_df = small_ts.loc[pix.isin(scenario=[f"scenario_{i}" for i in range(2)])].ct.to_sns_df(increase_resolution=100)
sns_df

# %%
sns.lineplot(
    data=sns_df[sns_df["time"] <= 1855],
    x="time",
    y="value",
    hue="scenario",
    style="variable",
    estimator=None,
    units="run",
)

# %% [markdown]
# - other operations, also with progress, parallel, parallel with progress
# - plot with basic control over labels
# - plot with grouping and plumes for ranges
# - convert with more fine-grained control over interpolation
#   (e.g. interpolation being passed as pd.Series)
# - unit conversion
