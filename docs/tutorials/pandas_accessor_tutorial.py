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

import matplotlib.pyplot as plt
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
        50.0
        * np.linspace(0.3, 1, n_ts)[:, np.newaxis]
        * np.linspace(0, 1, timepoints.size)[np.newaxis, :]
        + np.random.random((n_ts, timepoints.size)),
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
    interpolation=ct.InterpolationOption.Quadratic,
)
small_ts

# %%
small_ts.ct.differentiate(progress=True)

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
ax = small_ts.loc[pix.isin(variable="variable_0")].ct.plot(
    continuous_plot_kwargs=dict(alpha=0.3)
)
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))

# %%
# TODO: move this to plotting section
ax = small_ts.loc[pix.isin(variable="variable_0", run=0)].ct.plot(
    label="scenario", continuous_plot_kwargs=dict(alpha=0.9)
)
ax.legend()

# %%
# TODO: move this to ops section
ax = (
    small_ts.loc[pix.isin(variable="variable_0", run=0)]
    .ct.differentiate()
    .ct.plot(label="scenario", continuous_plot_kwargs=dict(alpha=0.9))
)
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
bigger_ts = bigger_df.ct.to_timeseries(
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
bigger_ts

# %% [markdown]
# The same logic can be applied to other operations.

# %%
diff_ts = create_df(
    n_scenarios=50,
    n_variables=1,
    n_runs=600,
    timepoints=np.arange(75) + 2025.0,
).ct.to_timeseries(
    time_units="yr",
    interpolation=ct.InterpolationOption.Linear,
    n_processes=n_processes,
    progress=True,
    progress_nested=True,
    mp_context=multiprocessing.get_context("fork"),
)
diff_ts

# %%
diff_ts.ct.differentiate(progress=True)

# %%
diff_ts.ct.differentiate(n_processes=n_processes)

# %%
diff_ts.ct.differentiate(n_processes=n_processes, progress=True)

# %%
diff_ts.ct.differentiate(
    n_processes=n_processes,
    progress=True,
    progress_nested=True,
    mp_context=multiprocessing.get_context("fork"),
)

# %% [markdown]
# Demonstrate how to control parallel etc. with global config.

# %% [markdown]
# On big `pd.DataFrame`'s the combination with
# [pandas indexing](https://pandas-indexing.readthedocs.io/)
# becomes particularly powerful.

# %%
ax = (
    bigger_df.loc[pix.isin(variable="variable_1")]
    .groupby(bigger_df.index.names.difference(["run"]), observed=True)
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
# pd.testing.assert_frame_equal(
#     small_df,
#     # Units don't round trip by default
#     small_ts.ct.to_df(out_units="Mt / yr")
# )
small_ts.ct

# %%
small_ts.ct.to_df(increase_resolution=3)

# %%
sns_df = small_ts.loc[
    pix.isin(scenario=[f"scenario_{i}" for i in range(2)])
    # Rename to `to_tidy_df`
].ct.to_sns_df(increase_resolution=100)
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

# %%
from itertools import cycle

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

fig, ax = plt.subplots()
in_ts = small_ts.loc[pix.isin(variable="variable_0")]
quantile_over = "run"
pre_calculated = False
observed = True
quantiles_plumes = (
    ((0.5,), 0.8),
    ((0.25, 0.75), 0.75),
    ((0.05, 0.95), 0.5),
)
hue_var = "scenario"
hue_var_label = None
style_var = "variable"
style_var_label = None
palette = None
dashes = None
observed = True
increase_resolution = 100
linewidth = 2

# The joy of plotting, you create everything yourself.
# TODO: split creation from use?
if hue_var_label is None:
    hue_var_label = hue_var.capitalize()
if style_var_label is None:
    style_var_label = style_var.capitalize()

quantiles = []
for quantile_plot_def in quantiles_plumes:
    q_def = quantile_plot_def[0]
    try:
        for q in q_def:
            quantiles.append(q)
    except TypeError:
        quantiles.append(q_def)

_palette = {} if palette is None else palette

if dashes is None:
    _dashes = {}
    lines = ["-", "--", "-.", ":"]
    linestyle_cycler = cycle(lines)
else:
    _dashes = dashes

# Need to keep track of this, just in case we end up plotting only plumes
_plotted_lines = False

quantile_labels = {}
plotted_hues = []
plotted_styles = []
units_l = []
for q, alpha in quantiles_plumes:
    for hue_value, hue_ts in in_ts.groupby(hue_var, observed=observed):
        for style_value, hue_style_ts in hue_ts.groupby(style_var, observed=observed):
            # Remake in inner loop to avoid leaking between plots
            pkwargs = {"alpha": alpha}

            if pre_calculated:
                # Should add some checks here
                raise NotImplementedError()
                # Maybe something like the below
                # missing_quantile = False
                # for qt in q:
                #     if qt not in quantiles:
                #         warnings.warn(
                #             f"Quantile {qt} not available for {hue_value=} {style_value=}"
                #         )
                #         missing_quantile = True

                # if missing_quantile:
                #     continue
            else:
                _pdf = (
                    hue_ts.ct.to_df(increase_resolution=increase_resolution)
                    .ct.groupby_except(quantile_over)
                    .quantile(quantiles)
                    .ct.fix_index_name_after_groupby_quantile()
                )

            if hue_value not in plotted_hues:
                plotted_hues.append(hue_value)

            x_vals = _pdf.columns.values.squeeze()
            # Require ur for this to work
            # x_vals = get_plot_vals(
            #     self.time_axis.bounds,
            #     "self.time_axis.bounds",
            #     warn_if_magnitudes=warn_if_plotting_magnitudes,
            # )

            if palette is not None:
                try:
                    pkwargs["color"] = _palette[hue_value]
                except KeyError:
                    error_msg = f"{hue_value} not in palette. {palette=}"
                    raise KeyError(error_msg)
            elif hue_value in _palette:
                pkwargs["color"] = _palette[hue_value]
            # else:
            #     # Let matplotlib default cycling do its thing

            n_q_for_plume = 2
            plot_plume = len(q) == n_q_for_plume
            plot_line = len(q) == 1

            if plot_plume:
                label = f"{q[0] * 100:.0f}th - {q[1] * 100:.0f}th"

                y_lower_vals = _pdf.loc[pix.ismatch(quantile=q[0])].values.squeeze()
                y_upper_vals = _pdf.loc[pix.ismatch(quantile=q[1])].values.squeeze()
                # Require ur for this to work
                # Also need the 1D check back in too
                # y_lower_vals = get_plot_vals(
                #     self.time_axis.bounds,
                #     "self.time_axis.bounds",
                #     warn_if_magnitudes=warn_if_plotting_magnitudes,
                # )
                p = ax.fill_between(
                    x_vals,
                    y_lower_vals,
                    y_upper_vals,
                    label=label,
                    **pkwargs,
                )

                if palette is None:
                    _palette[hue_value] = p.get_facecolor()[0]

            elif plot_line:
                if style_value not in plotted_styles:
                    plotted_styles.append(style_value)

                _plotted_lines = True

                if dashes is not None:
                    try:
                        pkwargs["linestyle"] = _dashes[style_value]
                    except KeyError:
                        error_msg = f"{style_value} not in dashes. {dashes=}"
                        raise KeyError(error_msg)
                else:
                    if style_value not in _dashes:
                        _dashes[style_value] = next(linestyle_cycler)

                    pkwargs["linestyle"] = _dashes[style_value]

                if isinstance(q[0], str):
                    label = q[0]
                else:
                    label = f"{q[0] * 100:.0f}th"

                y_vals = _pdf.loc[pix.ismatch(quantile=q[0])].values.squeeze()
                # Require ur for this to work
                # Also need the 1D check back in too
                # y_vals = get_plot_vals(
                #     self.time_axis.bounds,
                #     "self.time_axis.bounds",
                #     warn_if_magnitudes=warn_if_plotting_magnitudes,
                # )
                p = ax.plot(
                    x_vals,
                    y_vals,
                    label=label,
                    linewidth=linewidth,
                    **pkwargs,
                )[0]

                if dashes is None:
                    _dashes[style_value] = p.get_linestyle()

                if palette is None:
                    _palette[hue_value] = p.get_color()

            else:
                msg = f"quantiles to plot must be of length one or two, received: {q}"
                raise ValueError(msg)

            if label not in quantile_labels:
                quantile_labels[label] = p

            # Once we have unit handling with matplotlib, we can remove this
            # (and if matplotlib isn't set up, we just don't do unit handling)
            units_l.extend(_pdf.pix.unique("units").unique().tolist())

    # Fake the line handles for the legend
    hue_val_lines = [
        mlines.Line2D([0], [0], color=_palette[hue_value], label=hue_value)
        for hue_value in plotted_hues
    ]

    legend_items = [
        mpatches.Patch(alpha=0, label="Quantiles"),
        *quantile_labels.values(),
        mpatches.Patch(alpha=0, label=hue_var_label),
        *hue_val_lines,
    ]

    if _plotted_lines:
        style_val_lines = [
            mlines.Line2D(
                [0],
                [0],
                linestyle=_dashes[style_value],
                label=style_value,
                color="gray",
                linewidth=linewidth,
            )
            for style_value in plotted_styles
        ]
        legend_items += [
            mpatches.Patch(alpha=0, label=style_var_label),
            *style_val_lines,
        ]
    elif dashes is not None:
        warnings.warn(
            "`dashes` was passed but no lines were plotted, the style settings "
            "will not be used"
        )

    ax.legend(handles=legend_items, loc="best")

    if len(set(units_l)) == 1:
        ax.set_ylabel(units_l[0])

    # return ax, legend_items


quantiles

# %%
demo_q = (
    small_ts.ct.to_df(increase_resolution=5)
    .ct.groupby_except("run")
    .quantile([0.05, 0.5, 0.95])
    .ct.fix_index_name_after_groupby_quantile()
)
demo_q

# %%
units_col = "units"
indf = demo_q
out_l = []

# The 'shortcut'
target_units = "Gt / yr"
locs_target_units = ((pix.ismatch(**{units_col: "**"}), target_units),)
locs_target_units = (
    (pix.ismatch(scenario="scenario_2"), "Gt / yr"),
    (pix.ismatch(scenario="scenario_0"), "kt / yr"),
    (
        demo_q.index.get_level_values("scenario").isin(["scenario_1"])
        & demo_q.index.get_level_values("variable").isin(["variable_1"]),
        "t / yr",
    ),
)
# locs_target_units = (
#     (pix.ismatch(scenario="*"), "t / yr"),
# )

converted = None
for locator, target_unit in locs_target_units:
    if converted is None:
        converted = locator
    else:
        converted = converted | locator

    def _convert_unit(idf: pd.DataFrame) -> pd.DataFrame:
        start_units = idf.pix.unique(units_col).tolist()
        if len(start_units) > 1:
            msg = f"{start_units=}"
            raise AssertionError(msg)

        start_units = start_units[0]
        conversion_factor = UR.Quantity(1, start_units).to(target_unit).m

        return (idf * conversion_factor).pix.assign(**{units_col: target_unit})

    out_l.append(
        indf.loc[locator]
        .groupby(units_col, observed=True, group_keys=False)
        .apply(_convert_unit)
    )

out = pix.concat([*out_l, indf.loc[~converted]])
if isinstance(indf.index.dtypes[units_col], pd.CategoricalDtype):
    # Make sure that units stay as a category, if it started as one.
    out = out.reset_index(units_col)
    out[units_col] = out[units_col].astype("category")
    out = out.set_index(units_col, append=True).reorder_levels(indf.index.names)

out

# %% [markdown]
# - convert with more fine-grained control over interpolation
#   (e.g. interpolation being passed as pd.Series)
