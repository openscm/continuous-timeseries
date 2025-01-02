"""
Unit tests of `continuous_timeseries.timeseries_discrete`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import pint
import pint.testing
import pytest

from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.values_at_bounds import ValuesAtBounds

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "time_axis, values_at_bounds, expectation",
    (
        pytest.param(
            TimeAxis(Q([1750.0, 1850.0, 1950.0], "yr")),
            ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
            does_not_raise(),
            id="valid",
        ),
        pytest.param(
            TimeAxis(Q([1750.0, 1850.0, 1950.0, 2000.0], "yr")),
            ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values_at_bounds` must have values "
                    "that are the same shape as `self.time_axis.bounds`. "
                    "Received values_at_bounds.values.shape=(3,) "
                    "while self.time_axis.bounds.shape=(4,)."
                ),
            ),
            id="time_longer_than_values",
        ),
        pytest.param(
            TimeAxis(Q([1750.0, 1850.0], "yr")),
            ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values_at_bounds` must have values "
                    "that are the same shape as `self.time_axis.bounds`. "
                    "Received values_at_bounds.values.shape=(3,) "
                    "while self.time_axis.bounds.shape=(2,)."
                ),
            ),
            id="time_shorter_than_values",
        ),
    ),
)
def test_validation_time_axis_values_same_shape(
    time_axis, values_at_bounds, expectation
):
    with expectation:
        TimeseriesDiscrete(
            name="name", time_axis=time_axis, values_at_bounds=values_at_bounds
        )
