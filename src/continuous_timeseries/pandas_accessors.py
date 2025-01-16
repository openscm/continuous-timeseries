"""
Accessors for pandas
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pint

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.timeseries import Timeseries

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

    def to_timeseries(
        self,
        time_units: str | pint.facets.plain.PlainUnit,
        interpolation: InterpolationOption,
        unit_col: str = "units",
        ur: None = None,
        idx_separator: str = "__",
    ) -> Timeseries:
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
        x = self._df.columns.values * time_units
        # TODO: move to validation
        try:
            units_idx = self._df.index.names.index(unit_col)
        except ValueError as exc:
            msg = f"{unit_col} not available. {self._df.index.names=}"

            raise KeyError(msg) from exc

        ts_store = []
        for i, (idx, row) in enumerate(self._df.iterrows()):
            row_y = row.values * ur.Unit(idx[units_idx])

            if not isinstance(interpolation, InterpolationOption):
                # Would want the user to be able to pick interpolation by row
                # if they want (means allowing both passing a pd.Series of
                # InterpolationOption and some mapping between the index and
                # the interpolation function to use.
                raise NotImplementedError(interpolation)

            row_interpolation = interpolation

            # TODO: This is where injection of custom creators would need to be handled
            ts = Timeseries.from_arrays(
                x=x,
                y=row_y,
                interpolation=row_interpolation,
                name=idx_separator.join(idx),
            )
            ts_store.append((ts, idx))

        res = pd.Series(
            [v[0] for v in ts_store],
            index=pd.MultiIndex.from_tuples(
                [v[1] for v in ts_store],
                names=self._df.index.names,
            ),
            name="ts",
        )

        return res


def register_pandas_accessor() -> None:
    try:
        import pandas
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "register_pandas_accessor", requirement="pandas"
        ) from exc

    pandas.api.extensions.register_dataframe_accessor("ct")(DataFrameCTAccessor)
