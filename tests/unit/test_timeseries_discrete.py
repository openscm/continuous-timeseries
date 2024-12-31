"""
Test the `timeseries_discrete` module
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pint
import pint.testing
import pytest
from IPython.lib.pretty import pretty

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


formatting_check_cases = pytest.mark.parametrize(
    "ts",
    (
        pytest.param(
            TimeseriesDiscrete(
                name="basic",
                time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
                values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
            ),
            id="basic",
        ),
        pytest.param(
            TimeseriesDiscrete(
                name="big_array",
                time_axis=TimeAxis(Q(np.linspace(1750, 1850, 1000), "yr")),
                values_at_bounds=ValuesAtBounds(Q(np.arange(1000), "kg")),
            ),
            id="big_array",
        ),
        pytest.param(
            TimeseriesDiscrete(
                name="really_big_array",
                time_axis=TimeAxis(Q(np.linspace(1750, 1850, int(1e5)), "yr")),
                values_at_bounds=ValuesAtBounds(Q(np.arange(1e5), "kg")),
            ),
            id="really_big_array",
        ),
    ),
)


@formatting_check_cases
def test_repr(ts, file_regression):
    exp = (
        "TimeseriesDiscrete("
        f"name={ts.name!r}, "
        f"time_axis={ts.time_axis!r}, "
        f"values_at_bounds={ts.values_at_bounds!r}"
        ")"
    )

    assert repr(ts) == exp

    file_regression.check(
        f"{ts!r}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_str(ts, file_regression):
    exp = (
        "TimeseriesDiscrete("
        f"name={ts.name}, "
        f"time_axis={ts.time_axis}, "
        f"values_at_bounds={ts.values_at_bounds}"
        ")"
    )

    assert str(ts) == exp

    file_regression.check(
        f"{ts}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_pretty(ts, file_regression):
    file_regression.check(
        f"{pretty(ts)}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_html(ts, file_regression):
    file_regression.check(
        f"{ts._repr_html_()}\n",
        extension=".html",
    )


def test_plot():
    raise NotImplementedError()
