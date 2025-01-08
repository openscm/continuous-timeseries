"""
Integration tests of `continuous_timeseries.values_at_bounds`
"""

from __future__ import annotations

import textwrap

import numpy as np
import pint
import pytest

from continuous_timeseries.values_at_bounds import ValuesAtBounds

UR = pint.get_application_registry()
Q = UR.Quantity


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


@pytest.mark.parametrize(
    "values",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "kg"),
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
def test_pretty(values):
    pytest.importorskip("IPython")

    from IPython.lib.pretty import pretty

    formatted = f"values={pretty(values)}"

    if len(formatted) > 60:
        indented = textwrap.indent(formatted, "    ")
        exp = f"ValuesAtBounds(\n{indented})"
    else:
        exp = f"ValuesAtBounds({formatted})"

    instance = ValuesAtBounds(values)

    assert pretty(instance) == exp


@pytest.mark.parametrize(
    "values",
    (
        pytest.param(
            Q([1.0, 2.0, 3.0], "kg"),
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
def test_html(values, file_regression):
    instance = ValuesAtBounds(values)

    file_regression.check(
        f"{instance._repr_html_()}\n",
        extension=".html",
    )
