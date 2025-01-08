"""
Unit tests of `continuous_timeseries.timeseries_continuous`
"""

from __future__ import annotations

import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest

from continuous_timeseries.exceptions import (
    MissingOptionalDependencyError,
)
from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
    TimeseriesContinuous,
    get_plot_points,
)

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "domain, expectation",
    (
        pytest.param(
            (Q(1750, "yr"), Q(1850, "yr")),
            does_not_raise(),
            id="valid",
        ),
        pytest.param(
            (Q(1850, "yr"), Q(1750, "yr")),
            pytest.raises(ValueError),
            id="wrong_order",
        ),
        pytest.param(
            (Q(1850, "yr"),),
            pytest.raises(ValueError),
            id="wrong_size_too_short",
        ),
        pytest.param(
            (Q(1850, "yr"), Q(1750, "yr"), Q(1950, "yr")),
            pytest.raises(ValueError),
            id="wrong_size_too_long",
        ),
    ),
)
def test_validation_time_axis_values_same_shape(domain, expectation):
    with expectation:
        TimeseriesContinuous(
            name="name",
            time_units=UR.Unit("yr"),
            values_units=UR.Unit("Gt"),
            function="not_checked",
            domain=domain,
        )


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
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    continuous_function_scipy_ppoly = ContinuousFunctionScipyPPoly(
        scipy_interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            continuous_function_scipy_ppoly.integrate(0.0, domain_start=1.0)


@pytest.mark.parametrize(
    "time_axis, res_increase, exp",
    (
        pytest.param(
            Q([1750, 1850, 1900, 2000, 2010, 2030], "yr"),
            1,
            Q([1750, 1850, 1900, 2000, 2010, 2030], "yr"),
            id="res_increase_equal_1",
        ),
        pytest.param(
            Q([1750, 1850, 1900, 2000, 2010, 2030], "yr"),
            2,
            Q([1750, 1800, 1850, 1875, 1900, 1950, 2000, 2005, 2010, 2020, 2030], "yr"),
            id="res_increase_equal_2",
        ),
        pytest.param(
            Q([1750, 1780, 1840], "yr"),
            3,
            Q([1750, 1760, 1770, 1780, 1800, 1820, 1840], "yr"),
            id="res_increase_equal_3",
        ),
        pytest.param(
            Q([1750, 1751, 1763], "yr"),
            12,
            Q(
                np.hstack(
                    [
                        np.arange(1750, 1751, 1 / 12),
                        np.arange(1751, 1763.01, 1.0),
                    ]
                ),
                "yr",
            ),
            id="res_increase_equal_12",
        ),
    ),
)
def test_get_plot_points(time_axis, res_increase, exp):
    res = get_plot_points(time_axis, res_increase)

    pint.testing.assert_allclose(res, exp)

    # Check docs are correct
    docs_n_points = time_axis.size + (res_increase - 1) * (time_axis.size - 1)
    assert docs_n_points == res.size
