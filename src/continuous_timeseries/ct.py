"""
Accessors for pandas
"""

from __future__ import annotations

import pandas as pd


def validate(df: pd.DataFrame) -> None:
    """
    Validate the provided data can be used
    """


@pd.api.extensions.register_dataframe_accessor("ct")
class DataFrameCTAccessor:
    """
    [`pd.DataFrame`][pandas.DataFrame] accessors

    For details, see
    [pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        validate(pandas_obj)
        self._obj = pandas_obj

    @property
    def data(self) -> pd.DataFrame:
        """
        Get data
        """
        return self._obj

    def to_continuous(self) -> None:
        """
        To continuous timeseries
        """
        raise NotImplementedError
