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
class SeriesCTAccessor:
    """
    [`pd.Series`][pandas.Series] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.Series):
        """
        Initialise

        Parameters
        ----------
        pandas_obj
            Pandas object to use via the accessors
        """
        # TODO: add validation
        # validate(pandas_obj)
        self._series = pandas_obj

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Get the metadata
        """
        return self._series.index.to_frame(index=False)


# %%
import pandas
namespace = "ct"
pandas.api.extensions.register_series_accessor(namespace)(SeriesCTAccessor)

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
series = df.ct.to_timeseries(
    time_units=x.units,
    # interpolation=InterpolationOption.PiecewiseConstantPreviousLeftClosed
    interpolation=InterpolationOption.Linear
)
series

# %%
series.ct.metadata

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
import concurrent.futures
import multiprocessing


# %%
def get_executor_and_futures(
    in_iter,
    func_to_call,
    n_processes: int,
    mp_context = None,
    progress: bool = False,
    *args,
    **kwargs,
):
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "to_timeseries", requirement="pandas"
            ) from exc

        iterator = tqdm(in_iter, desc="submitting to parallel executor")
            
    else:
        iterator = in_iter
        
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=n_processes, mp_context=mp_context
    )

    futures = tuple(
        executor.submit(
            func_to_call,
            inv,
            *args,
            **kwargs,
        )
        for inv in iterator
    )

    return executor, futures


# %%
from continuous_timeseries import Timeseries

def get_ts(inv, *args, **kwargs):
    return inv[0], Timeseries.from_pandas_iterrows_value(inv, *args, **kwargs)
    


def to_timeseries(
    self,
    time_units, #: str | pint.facets.plain.PlainUnit,
    interpolation, #: InterpolationOption,
    units_col: str = "units",
    ur: None = None,
    idx_separator: str = "__",
    progress: bool = False,
    n_processes: int = 1,
    mp_context = None,
):
    """
    Convert to [`Timeseries`][(p)]

    TODO: add parameters here

    Returns
    -------
    :
        Timeseries representation of the [`pd.DataFrame`][pandas.DataFrame]
    """
    # Late import to avoid hard dependency on pandas
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "to_timeseries", requirement="pandas"
        ) from exc

    if isinstance(time_units, str):
        raise NotImplementedError

    if ur is None:
        ur = pint.get_application_registry()

    # TODO: parallelise this
    # df = self._df
    df = self
    x = df.columns.values * time_units
    # TODO: move to validation
    try:
        units_idx = df.index.names.index(units_col)
    except ValueError as exc:
        msg = f"{units_col} not available. {df.index.names=}"

        raise KeyError(msg) from exc

    if n_processes == 1:
        iterator = df.iterrows()
            
    else:
        units_idx = df.index.names.index(units_col)
        executor, futures = get_executor_and_futures(
            tuple(v for v in df.iterrows()),
            get_ts,
            n_processes=n_processes,
            mp_context=mp_context,
            progress=progress,   
            interpolation=InterpolationOption.Linear,
            units_idx=units_idx,
            time_units="yr",
        )
        iterator = concurrent.futures.as_completed(futures)

    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "to_timeseries", requirement="pandas"
            ) from exc

        iterator = tqdm(iterator, desc="rows", total=df.shape[0])

    if n_processes == 1:
        res = tuple(
            get_ts(
                v,
                interpolation=InterpolationOption.Linear,
                units_idx=units_idx,
                time_units="yr",
            )
            for v in iterator
            )

    else:
        try:
            res = tuple(future.result() for future in iterator)
        finally:
            executor.shutdown()
        
    res = pd.Series(
        (v[1] for v in res),
        pd.MultiIndex.from_tuples((v[0] for v in res), names=df.index.names),
        name="ts",
    )

    return res


# %%
from tqdm.auto import tqdm
for _ in tqdm(df.iterrows()):
    pass

# %%
to_timeseries(df, time_units=x.units, interpolation=InterpolationOption.Linear, progress=True, n_processes=16, mp_context=multiprocessing.get_context("spawn"))

# %%
to_timeseries(df, time_units=x.units, interpolation=InterpolationOption.Linear, progress=True, n_processes=16, mp_context=multiprocessing.get_context("fork"))

# %%
to_timeseries(df, time_units=x.units, interpolation=InterpolationOption.Linear, progress=True, n_processes=1)

# %%
to_timeseries(df, time_units=x.units, interpolation=InterpolationOption.Linear, n_processes=4)

# %%
to_timeseries(df, time_units=x.units, interpolation=InterpolationOption.Linear, n_processes=1)

# %%
