"""
Unit tests of `continuous_timeseries.plotting_helpers`
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pint
import pint.testing
import pytest

from continuous_timeseries.plotting_helpers import get_plot_vals

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="matplotlib_available"),
        pytest.param(
            {"matplotlib": None},
            pytest.warns(
                UserWarning,
                match=re.escape(
                    "Could not import `matplotlib.units` "
                    "to set up unit-aware plotting. "
                    "We will simply try plotting magnitudes instead."
                ),
            ),
            id="matplotlib_not_available",
        ),
    ),
)
def test_plot_no_matplotlib_units(sys_modules_patch, expectation):
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            get_plot_vals(Q([1, 2, 3], "yr"), desc="values", warn_if_magnitudes=False)
