"""
Test the `timeseries_discrete` module
"""

from __future__ import annotations

import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest
from IPython.lib.pretty import pretty

from continuous_timeseries.exceptions import MissingOptionalDependencyError
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


@pytest.mark.parametrize(
    "plot_kwargs, legend",
    (
        pytest.param({}, False, id="defaults"),
        pytest.param({}, True, id="legend"),
        pytest.param(
            dict(label="custom"),
            True,
            id="label-overwriting",
        ),
        pytest.param(dict(set_xlabel=True), False, id="xlabel"),
        pytest.param(dict(set_ylabel=True), False, id="ylabel"),
        pytest.param(
            dict(x_units="months", y_units="Mt / yr", set_xlabel=True, set_ylabel=True),
            False,
            id="label-user-units",
        ),
        pytest.param(
            dict(marker="x", color="tab:green", label="demo"),
            True,
            id="kwargs-passing",
        ),
    ),
)
def test_plot(plot_kwargs, legend, image_regression, tmp_path):
    import matplotlib

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ts = TimeseriesDiscrete(
        name="test_plot",
        time_axis=TimeAxis(
            Q(
                [
                    1750.0,
                    1950.0,
                    1975.0,
                    2000.0,
                    2010.0,
                    2020.0,
                    2030.0,
                    2050.0,
                    2100.0,
                    2200.0,
                    2300.0,
                ],
                "yr",
            )
        ),
        values_at_bounds=ValuesAtBounds(
            Q(
                [0.0, 2.3, 6.4, 10.0, 11.0, 12.3, 10.2, 0.0, -5.0, -2.0, 0.3],
                "Gt / yr",
            )
        ),
    )

    ts.plot(ax=ax, **plot_kwargs)
    if legend:
        ax.legend()

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    image_regression.check(out_file.read_bytes())


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="matplotlib_available"),
        pytest.param(
            {"matplotlib": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match="`TimeseriesDiscrete.plot` requires matplotlib to be installed",
            ),
            id="matplotlib_not_available",
        ),
    ),
)
def test_plot_ax_creation(sys_modules_patch, expectation):
    ts = TimeseriesDiscrete(
        name="basic",
        time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            ts.plot()
