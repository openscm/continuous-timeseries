"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from __future__ import annotations

import pandas as pd
import pytest

import continuous_timeseries.pandas_accessors


@pytest.fixture()
def setup_pandas_accessor() -> None:
    # Not parallel safe, but good enough
    continuous_timeseries.pandas_accessors.register_pandas_accessor()

    yield None

    # Surprising and a bit annoying that there isn't a safer way to do this
    pd.Series._accessors.discard("ct")
    if hasattr(pd.Series, "ct"):
        del pd.Series.ct
