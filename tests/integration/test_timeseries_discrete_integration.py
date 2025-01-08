"""
Integration tests of `continuous_timeseries.timeseries_discrete`
"""

from __future__ import annotations

import sys
import warnings
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pint
import pint.testing
import pytest

from continuous_timeseries.discrete_to_continuous import InterpolationOption
from continuous_timeseries.exceptions import MissingOptionalDependencyError
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.values_at_bounds import ValuesAtBounds

UR = pint.get_application_registry()
Q = UR.Quantity


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
    pytest.importorskip("IPython")

    from IPython.lib.pretty import pretty

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


def test_to_continuous_timeseries_warning_suppression():
    start = TimeseriesDiscrete(
        name="name",
        time_axis=TimeAxis(Q([1, 2, 3], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10, 20, 30], "kg")),
    )

    # Make sure no warning is raised.
    # See https://docs.pytest.org/en/7.0.x/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start.to_continuous_timeseries(
            InterpolationOption.PiecewiseConstantPreviousLeftOpen,
            warn_if_output_values_at_bounds_could_confuse=False,
        )


@pytest.mark.parametrize(
    "x_units, y_units, plot_kwargs, legend",
    (
        pytest.param(None, None, {}, False, id="no-units-set"),
        pytest.param("month", None, {}, False, id="x-units-set"),
        pytest.param(None, "t / yr", {}, False, id="y-units-set"),
        pytest.param("s", "Gt / yr", {}, False, id="x-and-y-units-set"),
        pytest.param(None, None, {}, True, id="default-labels"),
        pytest.param(
            None, None, dict(label="overwritten"), True, id="overwrite-labels"
        ),
        pytest.param(
            "yr",
            "Gt / yr",
            dict(alpha=0.7, s=130),
            False,
            id="x-and-y-units-set-kwargs",
        ),
    ),
)
def test_plot(  # noqa: PLR0913
    x_units, y_units, plot_kwargs, legend, image_regression, tmp_path
):
    matplotlib = pytest.importorskip("matplotlib")

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.units

    # Setup matplotlib to use units
    UR.setup_matplotlib(enable=True)

    fig, ax = plt.subplots()

    mt_month = TimeseriesDiscrete(
        name="Mt per month",
        time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Mt / month") * 100.0),
    )

    gt_yr = TimeseriesDiscrete(
        name="Gt per year",
        time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Gt / yr")),
    )

    mt = TimeseriesDiscrete(
        name="Mt",
        time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Mt")),
    )

    if x_units is not None:
        ax.set_xlabel(x_units)
        ax.xaxis.set_units(x_units)

    if y_units is not None:
        ax.set_ylabel(y_units)
        ax.yaxis.set_units(y_units)

    # Even though timeseries are in different units,
    # use of pint with matplotib will ensure sensible units on plot.
    mt_month.plot(ax=ax, **plot_kwargs)
    gt_yr.plot(ax=ax, **plot_kwargs)

    # Trying to plot something with incompatible units will raise.
    with pytest.raises(matplotlib.units.ConversionError):
        mt.plot(ax=ax, **plot_kwargs)

    if legend:
        ax.legend()

    fig.tight_layout()

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Ensure we tear down
    plt.close()
    UR.setup_matplotlib(enable=False)


@pytest.mark.parametrize(
    "plot_kwargs, expectation",
    (
        pytest.param(
            {},
            pytest.warns(
                UserWarning,
                match=(
                    "The magnitude will be plotted "
                    "without any consideration of units"
                ),
            ),
            id="defaults",
        ),
        pytest.param(
            dict(warn_if_plotting_magnitudes=True),
            pytest.warns(
                UserWarning,
                match=(
                    "The magnitude will be plotted "
                    "without any consideration of units"
                ),
            ),
            id="warning",
        ),
        pytest.param(
            dict(warn_if_plotting_magnitudes=False),
            does_not_raise(),
            id="no-warning",
        ),
    ),
)
def test_plot_matplotlib_units_not_registered(
    plot_kwargs, expectation, image_regression, tmp_path
):
    matplotlib = pytest.importorskip("matplotlib")

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

    with expectation:
        ts.plot(ax=ax, **plot_kwargs)

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    plt.close()

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param(
            {},
            does_not_raise(),
            id="matplotlib_available",
        ),
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
    (pytest.importorskip("matplotlib"),)

    ts = TimeseriesDiscrete(
        name="basic",
        time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
        values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            ts.plot(warn_if_plotting_magnitudes=False)
