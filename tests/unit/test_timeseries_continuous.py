"""
Unit tests of `continuous_timeseries.timeseries_continuous`
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pint
import pint.testing
import pytest
import scipy.interpolate

from continuous_timeseries.exceptions import (
    MissingOptionalDependencyError,
)
from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
)

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="scipy_available"),
        pytest.param(
            {"scipy": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match=(
                    "`ContinuousFunctionScipyPPoly.integrate` "
                    "requires scipy to be installed"
                ),
            ),
            id="scipy_not_available",
        ),
    ),
)
def test_integrate_no_scipy(sys_modules_patch, expectation):
    continuous_function_scipy_ppoly = ContinuousFunctionScipyPPoly(
        scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            continuous_function_scipy_ppoly.integrate(0.0)
