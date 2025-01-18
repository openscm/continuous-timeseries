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
import pint

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
                    [f"variable_{i}" for i in range(n_variables)],
                    [f"scenario_{i}" for i in range(n_scenarios)],
                    [i for i in range(n_runs)],
                )
            ),
            columns=["scenario", "variable", "region", "units"],
            # This makes updates later way way faster
            dtype="category",
        )
    )

    df = pd.DataFrame(
        np.random.random((n_variables * n_runs * n_scenarios, timepoints.size)),
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
    n_scenarios=25,
    n_variables=10,
    n_runs=30,
    timepoints=np.arange(250) + 1850.0,
)
small_df

# %% [markdown]
# Then we convert it time series.

# %%
small_df.ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
)

# %% [markdown]
# Then we can use standard Continuous timeseries APIs,
# e.g. plotting.

# %%

# %% [markdown]
# If we have a bigger `pd.DataFrame`, this process can be much slower.
# If you're not sure what's happening, you can activate the progress bar if you have
# [`tdqm`](https://tqdm.github.io/) installed.

# %%
bigger_df = create_df(
    n_scenarios=100,
    n_variables=2,
    n_runs=300,
    timepoints=np.arange(351) + 1850.0,
)
bigger_df

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
# because of the need for forking...).

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
# - filtering with pandas-indexing
# - bigger df
# - convert more rows (progress, parallel, parallel with progress)
# - other operations, also with progress, parallel, parallel with progress
# - convert to seaborn df for more fine-grained plotting control
#   - also requires adding a `increase_resolution` method to `Timeseries`
# - convert with more fine-grained control over interpolation
# - unit conversion
