"""
Accessors for pandas
"""

from __future__ import annotations

import pandas as pd

from continuous_timeseries.timeseries import Timeseries


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


@pd.api.extensions.register_dataframe_accessor("ct")
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

    def to_timeseries(self) -> Timeseries:
        """
        Convert to [`Timeseries`][(p)]

        Returns
        -------
        :
            Timeseries representation of the [`pd.DataFrame`][pandas.DataFrame]
        """
        raise NotImplementedError
