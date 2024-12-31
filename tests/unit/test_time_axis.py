"""
Test the `time_axis` module
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

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "bounds, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0], dtype=np.float32), "yr"),
            does_not_raise(),
            id="pint_1d_numpy_float_array",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "month"),
            does_not_raise(),
            id="pint_1d_numpy_int_array",
        ),
        pytest.param(
            Q(1.0, "yr"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional "
                    "but an error was raised while trying to check its shape. "
                    "Received bounds=1.0 year."
                ),
            ),
            id="pint_scalar",
        ),
        pytest.param(
            Q([1.0, 2.0, 3.0], "yr"),
            does_not_raise(),
            id="pint_1d_list",
        ),
        pytest.param(
            Q(np.array([[1.0, 2.0], [3.0, 4.0]]), "hour"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional. "
                    "Received `bounds` with shape (2, 2)"
                ),
            ),
            id="pint_2d_numpy_array",
        ),
        pytest.param(
            Q([[1.0, 2.0], [3.0, 4.0]], "s"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`bounds` must be one-dimensional. "
                    "Received `bounds` with shape (2, 2)"
                ),
            ),
            id="pint_2d_list",
        ),
    ),
)
def test_validation_shape(bounds, expectation):
    with expectation:
        TimeAxis(bounds)


@pytest.mark.parametrize(
    "bounds, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0]), "yr"),
            does_not_raise(),
            id="valid",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "month"),
            does_not_raise(),
            id="valid_int",
        ),
        pytest.param(
            Q(np.array([-1.0, 0.0, 1.0]), "yr"),
            does_not_raise(),
            id="valid_negative_numbers",
        ),
        pytest.param(
            Q(np.array([1.0, 2.0, 1.5]), "yr"),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`bounds` must be strictly monotonically increasing. "
                    "Received bounds=[1.0 2.0 1.5] year"
                ),
            ),
            id="invalid_decreasing",
        ),
        pytest.param(
            Q(np.array([1.0, 2.0, 2.0]), "yr"),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`bounds` must be strictly monotonically increasing. "
                    "Received bounds=[1.0 2.0 2.0] year"
                ),
            ),
            id="invalid_constant",
        ),
    ),
)
def test_validation_monotonically_increasing(bounds, expectation):
    with expectation:
        TimeAxis(bounds)


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


@pytest.mark.parametrize(
    "bounds, exp",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0]), "yr"),
            Q(np.array([[1.0, 2.0], [2.0, 3.0]]), "yr"),
            id="basic",
        ),
        pytest.param(
            Q(np.array([1.0, 10.0, 30.0]), "yr"),
            Q(np.array([[1.0, 10.0], [10.0, 30.0]]), "yr"),
            id="uneven_spacing",
        ),
    ),
)
def test_bounds2d(bounds, exp):
    res = TimeAxis(bounds).bounds_2d

    pint.testing.assert_equal(res, exp)
