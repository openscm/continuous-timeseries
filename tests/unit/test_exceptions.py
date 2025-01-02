"""
Unit tests of `continuous_timeseries.exceptions`
"""

from __future__ import annotations

import re

import pytest

from continuous_timeseries.exceptions import MissingOptionalDependencyError


def test_missing_optional_dependency_error():
    with pytest.raises(
        MissingOptionalDependencyError,
        match=re.escape("`module.function` requires seaborn to be installed"),
    ):
        raise MissingOptionalDependencyError(
            callable_name="module.function", requirement="seaborn"
        )
