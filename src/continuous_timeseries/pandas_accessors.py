"""
Accessors for pandas
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Callable, TypeVar

import pint
from typing_extensions import Concatenate, ParamSpec

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries import Timeseries
from continuous_timeseries.typing import PINT_NUMPY_ARRAY

if TYPE_CHECKING:
    import pandas as pd
    import pint.facets.plain


def validate(df: pd.DataFrame) -> None:
    """
    Validate the provided data can be used

    Parameters
    ----------
    df
        Data to validate

    Raises
    ------
    CTAccessorUnsupportedError
        `df` is not supported by continuous timeseries' pandas accessors.
    """


P = ParamSpec("P")
T = TypeVar("T")
U = TypeVar("U")


def get_executor_and_futures(
    in_iter: Iterable[U],
    func_to_call: Callable[Concatenate[U, P], T],
    n_processes: int,
    mp_context: BaseContext | None = None,
    progress: bool = False,
    *args,
    **kwargs,
) -> tuple[T, ...]:
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


def interpolate_parallel_helper(
    in_v: tuple[tuple[str | float | int, ...], Timeseries],
    time_axis: TimeAxis | PINT_NUMPY_ARRAY,
    allow_extrapolation: bool = False,
) -> Timeseries:
    return in_v[0], in_v[1].interpolate(
        time_axis=time_axis, allow_extrapolation=allow_extrapolation
    )


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
        # validate_series(pandas_obj)
        self._series = pandas_obj

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Get the metadata
        """
        return self._series.index.to_frame(index=False)

    def interpolate(
        self,
        time_axis: TimeAxis | PINT_NUMPY_ARRAY,
        allow_extrapolation: bool = False,
        progress: bool = False,
        n_processes: int = 1,
        mp_context: BaseContext | None = None,
    ) -> pd.Series[Timeseries]:
        # Late import to avoid hard dependency on pandas
        try:
            import pandas as pd
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "interpolate", requirement="pandas"
            ) from exc

        if n_processes == 1:
            if progress:
                try:
                    from tqdm.auto import tqdm
                except ImportError as exc:
                    raise MissingOptionalDependencyError(  # noqa: TRY003
                        "interpolate(..., progress=True)", requirement="tdqm"
                    ) from exc

                tqdm.pandas(desc="timeseries")
                res = self._series.progress_map(
                    lambda x: x.interpolate(
                        time_axis=time_axis,
                        allow_extrapolation=all,
                    )
                )

            else:
                res = self._series.map(
                    lambda x: x.interpolate(
                        time_axis=time_axis,
                        allow_extrapolation=all,
                    )
                )

            return res

        executor, futures = get_executor_and_futures(
            tuple(v for v in self._series.items()),
            interpolate_parallel_helper,
            n_processes=n_processes,
            mp_context=mp_context,
            progress=progress,
            time_axis=time_axis,
            allow_extrapolation=all,
        )
        iterator = concurrent.futures.as_completed(futures)

        if progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise MissingOptionalDependencyError(  # noqa: TRY003
                    "interpolate(..., progress=True)", requirement="tqdm"
                ) from exc

            iterator = tqdm(iterator, desc="timeseries", total=self._series.size)

        try:
            res_l = tuple(future.result() for future in iterator)
        finally:
            executor.shutdown()

        res = pd.Series(
            (v[1] for v in res_l),
            pd.MultiIndex.from_tuples(
                (v[0] for v in res_l), names=self._series.index.names
            ),
        )
        return res


def get_timeseries_parallel_helper(inv, units_idx: int, *args, **kwargs):
    return [
        v for i, v in enumerate(inv[0]) if i != units_idx
    ], Timeseries.from_pandas_iterrows_value(inv, units_idx=units_idx, *args, **kwargs)


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
            Pandas object to use via the accessors
        """
        validate(pandas_obj)
        self._df = pandas_obj

    # # This is how you do a property, should we ever need it
    # @property
    # def data(self) -> pd.DataFrame:
    #     """
    #     Get data
    #     """
    #     return self._obj

    def to_timeseries(  # noqa: PLR0913
        self,
        time_units: str | pint.facets.plain.PlainUnit,
        interpolation: InterpolationOption,
        units_col: str = "units",
        ur: None = None,
        idx_separator: str = "__",
        progress: bool = False,
        n_processes: int = 1,
        mp_context: BaseContext | None = None,
    ) -> pd.Series[Timeseries]:
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

        df = self._df

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
                get_timeseries_parallel_helper,
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
                get_timeseries_parallel_helper(
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

        res_index_names = [v for v in df.index.names if v != units_col]
        res = pd.Series(
            (v[1] for v in res),
            pd.MultiIndex.from_tuples((v[0] for v in res), names=res_index_names),
            name="ts",
        )

        return res


def register_pandas_accessor(namespace: str = "ct") -> None:
    """
    Register the pandas accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

    Parameters
    ----------
    namespace
        Namespace to use for the accessor
    """
    # Doing this because I really don't like imports having side effects
    try:
        import pandas
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "register_pandas_accessor", requirement="pandas"
        ) from exc

    pandas.api.extensions.register_series_accessor(namespace)(SeriesCTAccessor)
    pandas.api.extensions.register_dataframe_accessor(namespace)(DataFrameCTAccessor)
