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
