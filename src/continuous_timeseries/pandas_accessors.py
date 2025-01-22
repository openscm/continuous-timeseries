"""
API for [`pandas`][pandas] accessors.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterator
from functools import partial
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pint

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.timeseries import Timeseries

if TYPE_CHECKING:
    import matplotlib
    import pandas as pd

    P = TypeVar("P", bound=pd.DataFrame | pd.Series[Any])


def apply_pandas_op_parallel(
    obj,
    op,
    n_processes: int,
    progress: bool = False,
    progress_nested: bool = False,
    mp_context: BaseContext | None = None,
):
    iterator = get_chunks(obj, n_chunks=n_processes)
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "apply_pandas_op_parallel(..., progress=True)", requirement="tdqm"
            ) from exc

        iterator = tqdm(iterator, desc="submitting to pool")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_processes, mp_context=mp_context
    ) as pool:
        futures = [
            pool.submit(
                op,
                chunk,
                progress=progress_nested,
                progress_bar_position=i,
            )
            for i, chunk in enumerate(iterator)
        ]

        iterator_results = concurrent.futures.as_completed(futures)
        if progress:
            iterator_results = tqdm(
                iterator_results,
                desc="Retrieving parallel results",
                total=len(futures),
            )

        res_l = [future.result() for future in iterator_results]

    # Late import to avoid hard dependency on pandas
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "apply_pandas_op_parallel", requirement="pandas"
        ) from exc

    # This assumes that the index isn't mangled.
    # Using pix.concat might be safer,
    # or we make the concatenation injectable.
    res = pd.concat(res_l)

    return res


def differentiate_parallel_helper(
    series: pd.Series[Timeseries],
    progress: bool = False,
    progress_bar_position: int = 0,
) -> pd.Series[Timeseries]:
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "dist(..., progress=True)", requirement="tdqm"
            ) from exc

        tqdm_kwargs = dict(position=progress_bar_position)
        tqdm.pandas(**tqdm_kwargs)
        meth_to_call = "progress_map"
        # No-one knows why this is needed, but it is in jupyter notebooks
        print(end=" ")

    else:
        meth_to_call = "map"

    res = getattr(series, meth_to_call)(
        lambda x: x.differentiate(),
        # name="injectable?",
    )

    return res


class SeriesCTAccessor:
    """
    [`pd.Series`][pandas.Series] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.Series[Timeseries]):  # type: ignore # pandas-stubs doesn't allow object even though it's fine
        """
        Initialise

        Parameters
        ----------
        pandas_obj
            Pandas object to use via the accessor
        """
        # TODO: consider adding validation
        # validate_series(pandas_obj)
        self._series = pandas_obj

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Get the metadata as a [`pd.DataFrame`][pandas.DataFrame]
        """
        return self._series.index.to_frame(index=False)

    def to_df(self, increase_resolution: int | None = None) -> pd.DataFrame:
        # Late import to avoid hard dependency on pandas
        try:
            import pandas as pd
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "to_pandas_series", requirement="pandas"
            ) from exc

        series_l = []
        indexes_l = []
        # TODO: progress bar and parallelisation
        for idx, value in self._series.items():
            # TODO: time_units and out_units passing
            if increase_resolution is not None:
                value_use = value.increase_resolution(increase_resolution)
            else:
                value_use = value

            pd_series = value_use.to_pandas_series()
            series_l.append(pd_series)
            indexes_l.append((*idx, pd_series.name))

        idx = pd.MultiIndex.from_frame(
            pd.DataFrame(
                indexes_l,
                columns=[*self._series.index.names, "units"],
                dtype="category",
            )
        )
        df = pd.DataFrame(
            series_l,
            index=idx,
        )

        return df

    # TODO: add this to DataFrame accessor to allow for time filtering in the middle
    def to_sns_df(self, increase_resolution: int = 100) -> pd.DataFrame:
        # TODO: progress bar and parallelisation
        # TODO: time_units and out_units passing
        return (
            self.to_df(increase_resolution=increase_resolution)
            # Will become `.ct.to_sns_df`
            .melt(
                var_name="time",
                ignore_index=False,
            )
            .reset_index()
        )

    def differentiate(
        self,
        # res_name: str = "ts",
        progress: bool = False,
        progress_nested: bool = False,
        n_processes: int = 1,
        mp_context: BaseContext | None = None,
    ) -> pd.Series[Timeseries]:  # type: ignore
        if n_processes == 1:
            res = differentiate_parallel_helper(
                self._series,
                progress=progress,
            )

            return res

        res = apply_pandas_op_parallel(
            self._series,
            op=differentiate_parallel_helper,
            n_processes=n_processes,
            progress=progress,
            progress_nested=progress_nested,
            mp_context=mp_context,
        )

        return res

    def groupby_except(
        self, non_groupers: str | list[str], observed: bool = True
    ) -> pd.core.groupby.generic.SeriesGroupBy:
        if isinstance(non_groupers, str):
            non_groupers = [non_groupers]

        return self._series.groupby(
            self._series.index.names.difference(non_groupers), observed=observed
        )

    def plot(
        self,
        label: str | tuple[str, ...] | None = None,
        show_continuous: bool = True,
        continuous_plot_kwargs: dict[str, Any] | None = None,
        show_discrete: bool = False,
        discrete_plot_kwargs: dict[str, Any] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        progress: bool = False,
    ) -> matplotlib.axes.Axes:
        iterator = self._series.items()
        if progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise MissingOptionalDependencyError(  # noqa: TRY003
                    "get_timeseries_parallel_helper(..., progress=True)",
                    requirement="tdqm",
                ) from exc

            iterator = tqdm(iterator, desc="Timeseries to plot")

        if label is not None:
            if isinstance(label, tuple):
                raise NotImplementedError()

            label_idx: int | None = get_index_level_idx(self._series, index_level=label)

        else:
            label_idx = None

        for idx, ts in iterator:
            if label_idx is not None:
                label = idx[label_idx]
                if "label" in continuous_plot_kwargs:
                    # clash (could just warn here instead)
                    raise KeyError

                continuous_plot_kwargs_use = continuous_plot_kwargs | dict(label=label)

            else:
                continuous_plot_kwargs_use = continuous_plot_kwargs

            ax = ts.plot(
                show_continuous=show_continuous,
                continuous_plot_kwargs=continuous_plot_kwargs_use,
                show_discrete=show_discrete,
                discrete_plot_kwargs=discrete_plot_kwargs,
                ax=ax,
            )

        return ax


def get_chunks(pd_obj: P, n_chunks: int) -> Iterator[P]:
    # Late import to avoid hard dependency on pandas
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "interpolate", requirement="pandas"
        ) from exc

    if isinstance(pd_obj, pd.DataFrame):
        total = pd_obj.shape[0]
    else:
        # Series
        total = pd_obj.size

    chunk_size = int(np.ceil(total / n_chunks))
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end >= total:
            end = None

        if isinstance(pd_obj, pd.DataFrame):
            yield pd_obj.iloc[start:end, :]
        else:
            yield pd_obj.iloc[start:end]


def get_index_level_idx(obj: pd.DataFrame | pd.Series, index_level: str) -> int:
    try:
        level_idx = obj.index.names.index(index_level)
    except ValueError as exc:
        msg = f"{index_level} not available. {obj.index.names=}"
        raise KeyError(msg) from exc

    return level_idx


def get_timeseries_parallel_helper(
    df: pd.DataFrame,
    interpolation: InterpolationOption,
    time_units: str | pint.facets.plain.PlainUnit,
    units_col: str,
    idx_separator: str,
    ur: pint.facets.PlainRegistry | None = None,
    progress: bool = False,
    progress_bar_position: int = 0,
) -> pd.Series[Timeseries]:
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise MissingOptionalDependencyError(  # noqa: TRY003
                "get_timeseries_parallel_helper(..., progress=True)", requirement="tdqm"
            ) from exc

        tqdm_kwargs = dict(position=progress_bar_position)
        tqdm.pandas(**tqdm_kwargs)
        meth_to_call = "progress_apply"
        # No-one knows why this is needed, but it is in jupyter notebooks
        print(end=" ")

    else:
        meth_to_call = "apply"

    units_idx = get_index_level_idx(df, index_level=units_col)

    res = getattr(df, meth_to_call)(
        # TODO: make this injectable too
        # This will also allow us to introduce an extra layer
        # to handle the case when interpolation is a Series,
        # rather than the same across all rows.
        Timeseries.from_pandas_series,
        axis="columns",
        interpolation=interpolation,
        units_idx=units_idx,
        time_units=time_units,
        # name="injectable?",
        idx_separator=idx_separator,
        ur=ur,
    )
    # Units now handled by timeseries
    res = res.reset_index(units_col, drop=True)

    return res


class DataFrameCTAccessor:
    """
    [`pd.DataFrame`][pandas.DataFrame] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialise

        Parameters
        ----------
        pandas_obj
            Pandas object to use via the accessor
        """
        # TODO: consider adding validation
        # validate_series(pandas_obj)
        self._df = pandas_obj

    def to_timeseries(  # noqa: PLR0913
        self,
        interpolation: InterpolationOption,
        time_units: str | pint.facets.plain.PlainUnit,
        units_col: str = "units",
        ur: None = None,
        idx_separator: str = "__",
        res_name: str = "ts",
        progress: bool = False,
        progress_nested: bool = False,
        n_processes: int = 1,
        mp_context: BaseContext | None = None,
    ) -> pd.Series[Timeseries]:
        if n_processes == 1:
            res = get_timeseries_parallel_helper(
                self._df,
                interpolation=interpolation,
                time_units=time_units,
                units_col=units_col,
                idx_separator=idx_separator,
                ur=ur,
                progress=progress,
            )

            return res

        res = apply_pandas_op_parallel(
            self._df,
            op=partial(
                get_timeseries_parallel_helper,
                interpolation=interpolation,
                time_units=time_units,
                units_col=units_col,
                idx_separator=idx_separator,
                ur=ur,
            ),
            n_processes=n_processes,
            progress=progress,
            progress_nested=progress_nested,
            mp_context=mp_context,
        )

        return res

    def groupby_except(
        self, non_groupers: str | list[str], observed: bool = True
    ) -> pd.core.groupby.generic.DataFrameGroupBy:
        if isinstance(non_groupers, str):
            non_groupers = [non_groupers]

        return self._df.groupby(
            self._df.index.names.difference(non_groupers), observed=observed
        )

    def fix_index_name_after_groupby_quantile(self) -> pd.DataFrame:
        # TODO: think about doing in place
        res = self._df.copy()
        res.index = res.index.rename({None: "quantile"})

        return res


def register_pandas_accessor(namespace: str = "ct") -> None:
    """
    Register the pandas accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

    We provide this as a separate function
    because we have had really bad experiences with imports having side effects
    and don't want to pass those on to our users.

    Parameters
    ----------
    namespace
        Namespace to use for the accessor
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "register_pandas_accessor", requirement="pandas"
        ) from exc

    pd.api.extensions.register_series_accessor(namespace)(SeriesCTAccessor)
    pd.api.extensions.register_dataframe_accessor(namespace)(DataFrameCTAccessor)
