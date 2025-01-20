"""
Overall integration tests of our pandas accessors
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from continuous_timeseries.exceptions import (
    MissingOptionalDependencyError,
)

pd = pytest.importorskip("pandas")


def test_import_does_not_register():
    # Make sure the accessor is not registered on import.
    # It's a bit annoying that there isn't a better way to do this.
    if hasattr(pd.Series, "ct"):
        del pd.Series.ct
        pd.Series._accessors.discard("ct")

    import continuous_timeseries.pandas_accessors

    assert not hasattr(pd.Series, "ct")
    assert "ct" not in pd.Series._accessors

    # Make sure that registering then brings this back
    continuous_timeseries.pandas_accessors.register_pandas_accessor()

    assert hasattr(pd.Series, "ct")

    pd.Series._accessors.discard("ct")
    del pd.Series.ct


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="pandas_available"),
        pytest.param(
            {"pandas": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match="`register_pandas_accessor` requires pandas to be installed",
            ),
            id="pandas_not_available",
        ),
    ),
)
def test_registering(sys_modules_patch, expectation):
    pytest.importorskip("pandas")

    import continuous_timeseries.pandas_accessors

    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            continuous_timeseries.pandas_accessors.register_pandas_accessor()

    # Tidy up
    pd.Series._accessors.discard("ct")
    if hasattr(pd.Series, "ct"):
        del pd.Series.ct
