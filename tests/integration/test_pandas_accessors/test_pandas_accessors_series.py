"""
Tests of our pandas series accessors
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def test_metadata(setup_pandas_accessor):
    index = pd.MultiIndex.from_tuples(
        (
            ("a", "b", "c"),
            ("d", "e", "f"),
        ),
        names=["variable", "region", "units"],
    )
    start = pd.Series(
        ["not", "used"],
        index=index,
    )

    pd.testing.assert_frame_equal(start.ct.metadata, index.to_frame(index=False))
