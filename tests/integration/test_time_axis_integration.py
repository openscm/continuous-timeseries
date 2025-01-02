"""
Integration tests of `continuous_timeseries.time_axis`
"""

from __future__ import annotations

import numpy as np
import pint
import pint.testing
import pytest
from IPython.lib.pretty import pretty

from continuous_timeseries.time_axis import TimeAxis

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "bounds, exp_repr",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            "TimeAxis(bounds=<Quantity([1. 2. 3.], 'year')>)",
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            # There must be some internal limit in numpy.
            # This still just prints out all bounds,
            # but the really big array doesn't.
            f"TimeAxis(bounds={Q(np.linspace(1750, 2000 + 1, 1000), 'yr')!r})",
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            "TimeAxis(bounds=<Quantity([1750.         1750.00251003 1750.00502005 ... 2000.99497995 2000.99748997\n 2001.        ], 'year')>)",  # noqa: E501
            id="really_big_array",
        ),
    ),
)
def test_repr(bounds, exp_repr):
    instance = TimeAxis(bounds)

    assert repr(instance) == exp_repr


@pytest.mark.parametrize(
    "bounds, exp_str",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            "TimeAxis(bounds=[1.0 2.0 3.0] year)",
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            # There must be some internal limit in numpy.
            # This still just prints out all bounds,
            # but the really big array doesn't.
            f"TimeAxis(bounds={Q(np.linspace(1750, 2000 + 1, 1000), 'yr')})",
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            "TimeAxis(bounds=[1750.0 1750.0025100251003 1750.0050200502005 ... 2000.9949799497995 2000.9974899748997 2001.0] year)",  # noqa: E501
            id="really_big_array",
        ),
    ),
)
def test_str(bounds, exp_str):
    instance = TimeAxis(bounds)

    assert str(instance) == exp_str


@pytest.mark.parametrize(
    "bounds, exp_pretty",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            "TimeAxis(bounds=<Quantity([1. 2. 3.], 'year')>)",
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            (
                "TimeAxis(\n"
                f"bounds={pretty(Q(np.linspace(1750, 2000 + 1, 1000), 'yr'))})"
            ),
            marks=pytest.mark.skip(reason="Too hard to predict indenting and slow"),
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            (
                "TimeAxis(\n"
                "    bounds=<Quantity([1750.         1750.00251003 1750.00502005 ... 2000.99497995 2000.99748997\n"  # noqa: E501
                "     2001.        ], 'year')>)"
            ),
            id="really_big_array",
        ),
    ),
)
def test_pretty(bounds, exp_pretty):
    instance = TimeAxis(bounds)

    assert pretty(instance) == exp_pretty


@pytest.mark.parametrize(
    "bounds",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            id="really_big_array",
        ),
    ),
)
def test_html(bounds, file_regression):
    instance = TimeAxis(bounds)

    file_regression.check(
        f"{instance._repr_html_()}\n",
        extension=".html",
    )
