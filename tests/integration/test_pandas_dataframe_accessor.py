"""
Test our pandas DataFrame accessors

In other words, test the `pd.DataFrame.ct` namespace.
This may need to be split into multiple files in future.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def test_to_timeseries():
    assert False
