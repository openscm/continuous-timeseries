"""
Test the `values_at_bounds` module
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pint
import pytest

from continuous_timeseries.values_at_bounds import ValuesAtBounds

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "values, expectation",
    (
        pytest.param(
            Q(np.array([1.0, 2.0, 3.0], dtype=np.float32), "m"),
            does_not_raise(),
            id="pint_1d_numpy_float_array",
        ),
        pytest.param(
            Q(np.array([1, 2, 3], dtype=np.int32), "m"),
            does_not_raise(),
            id="pint_1d_numpy_int_array",
        ),
        pytest.param(
            Q(1.0, "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional "
                    "but an error was raised while trying to check its shape. "
                    f"Received values={Q(1.0, 'm')}."
                ),
            ),
            id="pint_scalar",
        ),
        pytest.param(
            Q([1.0, 2.0, 3.0], "m"),
            does_not_raise(),
            id="pint_1d_list",
        ),
        pytest.param(
            Q(np.array([[1.0, 2.0], [3.0, 4.0]]), "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional. "
                    "Received `values` with shape (2, 2)"
                ),
            ),
            id="pint_2d_numpy_array",
        ),
        pytest.param(
            Q([[1.0, 2.0], [3.0, 4.0]], "m"),
            pytest.raises(
                AssertionError,
                match=re.escape(
                    "`values` must be one-dimensional. "
                    "Received `values` with shape (2, 2)"
                ),
            ),
            id="pint_2d_list",
        ),
    ),
)
def test_validation(values, expectation):
    with expectation:
        ValuesAtBounds(values)


@pytest.mark.parametrize(
    "values, exp_repr",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "kg"),
            "ValuesAtBounds(values=<Quantity([1. 2. 3.], 'kilogram')>)",
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            # There must be some internal limit in numpy.
            # This still just prints out all values,
            # but the really big array doesn't.
            f"ValuesAtBounds(values={Q(np.linspace(1750, 2000 + 1, 1000), 'yr')!r})",
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            "ValuesAtBounds(values=<Quantity([1750.         1750.00251003 1750.00502005 ... 2000.99497995 2000.99748997\n 2001.        ], 'year')>)",  # noqa: E501
            id="really_big_array",
        ),
    ),
)
def test_repr(values, exp_repr):
    instance = ValuesAtBounds(values)

    assert repr(instance) == exp_repr


@pytest.mark.parametrize(
    "values, exp_str",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "kg"),
            "ValuesAtBounds(values=[1.0 2.0 3.0] kilogram)",
            id="basic",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, 1000), "yr"),
            # There must be some internal limit in numpy.
            # This still just prints out all values,
            # but the really big array doesn't.
            f"ValuesAtBounds(values={Q(np.linspace(1750, 2000 + 1, 1000), 'yr')})",
            id="big_array",
        ),
        pytest.param(
            Q(np.linspace(1750, 2000 + 1, int(1e5)), "yr"),
            "ValuesAtBounds(values=[1750.0 1750.0025100251003 1750.0050200502005 ... 2000.9949799497995 2000.9974899748997 2001.0] year)",  # noqa: E501
            id="really_big_array",
        ),
    ),
)
def test_str(values, exp_str):
    instance = ValuesAtBounds(values)

    assert str(instance) == exp_str
