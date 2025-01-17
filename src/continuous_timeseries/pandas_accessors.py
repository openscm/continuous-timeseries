"""
API for [`pandas`][pandas] accessors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.timeseries import Timeseries

if TYPE_CHECKING:
    import pandas as pd


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
