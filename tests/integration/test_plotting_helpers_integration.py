"""
Integration tests of `continuous_timeseries.plotting_helpers`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pint
import pint.testing
import pytest

from continuous_timeseries.plotting_helpers import get_plot_vals

pytest.importorskip("matplotlib")

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "register_units, warn_if_plotting_magnitudes, expectation, inv, exp_res",
    (
        pytest.param(
            False,
            True,
            pytest.warns(
                UserWarning,
                match=re.escape(
                    "The units of `values` "
                    "are not registered with matplotlib. "
                    "The magnitude will be plotted without any consideration of units. "
                    "For docs on how to set up unit-aware plotting, see "
                    "[the stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html) "  # noqa: E501
                    "(at the time of writing, the latest version's docs were "
                    "[v0.24.4](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))."
                ),
            ),
            Q([1, 2, 3], "yr"),
            np.ndarray([1, 2, 3]),
            id="units-not-registered-warn",
        ),
        pytest.param(
            False,
            False,
            does_not_raise(),
            Q([1, 2, 3], "yr"),
            np.ndarray([1, 2, 3]),
            id="units-not-registered-no-warn",
        ),
        pytest.param(
            True,
            True,
            does_not_raise(),
            Q([1, 2, 3], "yr"),
            Q([1, 2, 3], "yr"),
            id="units-registered-warn",
        ),
        pytest.param(
            True,
            False,
            does_not_raise(),
            Q([1, 2, 3], "yr"),
            Q([1, 2, 3], "yr"),
            id="units-registered-no-warn",
        ),
    ),
)
def test_get_plot_vals(
    register_units, warn_if_plotting_magnitudes, expectation, inv, exp_res
):
    if register_units:
        # Setup matplotlib to use units
        UR.setup_matplotlib(enable=True)

    with expectation:
        res = get_plot_vals(
            inv,
            desc="values",
            warn_if_magnitudes=warn_if_plotting_magnitudes,
        )

    if isinstance(exp_res, UR.Quantity):
        assert isinstance(res, UR.Quantity)
        pint.testing.assert_equal(res, inv)

    else:
        assert isinstance(res, np.ndarray)
        np.testing.assert_equal(res, inv.m)

    if register_units:
        # Ensure we tear down
        UR.setup_matplotlib(enable=False)
