"""
Test the `timeseries_continuous` module
"""

from __future__ import annotations

import re

import numpy as np
import pint
import pint.testing
import pytest
import scipy.interpolate
from IPython.lib.pretty import pretty

from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
)

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "ts, exp_re",
    (
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
            ),
            re.compile(
                r"ContinuousFunctionScipyPPoly\(ppoly=<scipy.interpolate._interpolate.PPoly object at .*>\)"  # noqa: E501
            ),
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[-1.2, 2.3], [10, 12]])
            ),
            re.compile(
                r"ContinuousFunctionScipyPPoly\(ppoly=<scipy.interpolate._interpolate.PPoly object at .*>\)"  # noqa: E501
            ),
        ),
    ),
)
def test_repr_continuous_function_scipy_ppoly(ts, exp_re):
    repr_value = repr(ts)

    assert exp_re.fullmatch(repr_value)


@pytest.mark.parametrize(
    "ts, exp",
    (
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
            ),
            "0th order ContinuousFunctionScipyPPoly(ppoly=scipy.interpolate._interpolate.PPoly(c=[[10. 12.]], x=[ 1. 10. 20.]))",  # noqa: E501
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[-1.2, 2.3], [10, 12]])
            ),
            (
                "1st order ContinuousFunctionScipyPPoly(ppoly=scipy.interpolate._interpolate.PPoly(c=[[-1.2  2.3]\n"  # noqa: E501
                " [10.  12. ]], x=[ 1. 10. 20.]))"
            ),
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=np.arange(10001), c=np.arange(20000).reshape(2, 10000)
                )
            ),
            (
                "1st order ContinuousFunctionScipyPPoly(ppoly=scipy.interpolate._interpolate.PPoly(c=[[0.0000e+00 1.0000e+00 2.0000e+00 ... 9.9970e+03 9.9980e+03 9.9990e+03]\n"  # noqa: E501
                " [1.0000e+04 1.0001e+04 1.0002e+04 ... 1.9997e+04 1.9998e+04 1.9999e+04]], x=[0.000e+00 1.000e+00 2.000e+00 ... 9.998e+03 9.999e+03 1.000e+04]))"  # noqa: E501
            ),
            id="heaps_of_windows",
        ),
    ),
)
def test_str_continuous_function_scipy_ppoly(ts, exp):
    str_value = str(ts)

    assert str_value == exp


@pytest.mark.parametrize(
    "ts, exp",
    (
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
            ),
            "0th order ContinuousFunctionScipyPPoly(\n"
            "    ppoly=scipy.interpolate._interpolate.PPoly(\n"
            "        c=array([[10., 12.]]),\n"
            "        x=array([ 1., 10., 20.])))",
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[-1.2, 2.3], [10, 12]])
            ),
            (
                "1st order ContinuousFunctionScipyPPoly(\n"
                "    ppoly=scipy.interpolate._interpolate.PPoly(\n"
                "        c=array([[-1.2,  2.3],\n"
                "               [10. , 12. ]]),\n"
                "        x=array([ 1., 10., 20.])))"
            ),
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=np.arange(10001), c=np.arange(20000).reshape(2, 10000)
                )
            ),
            (
                "1st order ContinuousFunctionScipyPPoly(\n"
                "    ppoly=scipy.interpolate._interpolate.PPoly(\n"
                "        c=array([[0.0000e+00, 1.0000e+00, 2.0000e+00, ..., 9.9970e+03, 9.9980e+03,\n"  # noqa: E501
                "                9.9990e+03],\n"
                "               [1.0000e+04, 1.0001e+04, 1.0002e+04, ..., 1.9997e+04, 1.9998e+04,\n"  # noqa: E501
                "                1.9999e+04]], shape=(2, 10000)),\n"
                "        x=array([0.000e+00, 1.000e+00, 2.000e+00, ..., 9.998e+03, 9.999e+03,\n"  # noqa: E501
                "               1.000e+04], shape=(10001,))))"
            ),
            id="heaps_of_windows",
        ),
    ),
)
def test_pretty_continuous_function_scipy_ppoly(ts, exp):
    pretty_value = pretty(ts)

    assert pretty_value == exp


@pytest.mark.parametrize(
    "ts",
    (
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
            ),
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[-1.2, 2.3], [10, 12]])
            ),
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=np.arange(10001), c=np.arange(20000).reshape(2, 10000)
                )
            ),
            id="heaps_of_windows",
        ),
    ),
)
def test_html_continuous_function_scipy_ppoly(ts, file_regression):
    file_regression.check(
        f"{ts._repr_html_()}\n",
        extension=".html",
    )


# @pytest.mark.parametrize(
#     "time_axis, values_at_bounds, expectation",
#     (
#         pytest.param(
#             TimeAxis(Q([1750.0, 1850.0, 1950.0], "yr")),
#             ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
#             does_not_raise(),
#             id="valid",
#         ),
#         pytest.param(
#             TimeAxis(Q([1750.0, 1850.0, 1950.0, 2000.0], "yr")),
#             ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
#             pytest.raises(
#                 AssertionError,
#                 match=re.escape(
#                     "`values_at_bounds` must have values "
#                     "that are the same shape as `self.time_axis.bounds`. "
#                     "Received values_at_bounds.values.shape=(3,) "
#                     "while self.time_axis.bounds.shape=(4,)."
#                 ),
#             ),
#             id="time_longer_than_values",
#         ),
#         pytest.param(
#             TimeAxis(Q([1750.0, 1850.0], "yr")),
#             ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
#             pytest.raises(
#                 AssertionError,
#                 match=re.escape(
#                     "`values_at_bounds` must have values "
#                     "that are the same shape as `self.time_axis.bounds`. "
#                     "Received values_at_bounds.values.shape=(3,) "
#                     "while self.time_axis.bounds.shape=(2,)."
#                 ),
#             ),
#             id="time_shorter_than_values",
#         ),
#     ),
# )
# def test_validation_time_axis_values_same_shape(
#     time_axis, values_at_bounds, expectation
# ):
#     with expectation:
#         TimeseriesContinuous(
#             name="name", time_axis=time_axis, values_at_bounds=values_at_bounds
#         )
#
#
# formatting_check_cases = pytest.mark.parametrize(
#     "ts",
#     (
#         pytest.param(
#             TimeseriesContinuous(
#                 name="basic",
#                 time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
#                 values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
#             ),
#             id="basic",
#         ),
#         pytest.param(
#             TimeseriesContinuous(
#                 name="big_array",
#                 time_axis=TimeAxis(Q(np.linspace(1750, 1850, 1000), "yr")),
#                 values_at_bounds=ValuesAtBounds(Q(np.arange(1000), "kg")),
#             ),
#             id="big_array",
#         ),
#         pytest.param(
#             TimeseriesContinuous(
#                 name="really_big_array",
#                 time_axis=TimeAxis(Q(np.linspace(1750, 1850, int(1e5)), "yr")),
#                 values_at_bounds=ValuesAtBounds(Q(np.arange(1e5), "kg")),
#             ),
#             id="really_big_array",
#         ),
#     ),
# )
#
#
# @formatting_check_cases
# def test_repr(ts, file_regression):
#     exp = (
#         "TimeseriesContinuous("
#         f"name={ts.name!r}, "
#         f"time_axis={ts.time_axis!r}, "
#         f"values_at_bounds={ts.values_at_bounds!r}"
#         ")"
#     )
#
#     assert repr(ts) == exp
#
#     file_regression.check(
#         f"{ts!r}\n",
#         extension=".txt",
#     )
#
#
# @formatting_check_cases
# def test_str(ts, file_regression):
#     exp = (
#         "TimeseriesContinuous("
#         f"name={ts.name}, "
#         f"time_axis={ts.time_axis}, "
#         f"values_at_bounds={ts.values_at_bounds}"
#         ")"
#     )
#
#     assert str(ts) == exp
#
#     file_regression.check(
#         f"{ts}\n",
#         extension=".txt",
#     )
#
#
# @formatting_check_cases
# def test_pretty(ts, file_regression):
#     file_regression.check(
#         f"{pretty(ts)}\n",
#         extension=".txt",
#     )
#
#
# @formatting_check_cases
# def test_html(ts, file_regression):
#     file_regression.check(
#         f"{ts._repr_html_()}\n",
#         extension=".html",
#     )
#
#
# @pytest.mark.parametrize(
#     "x_units, y_units, plot_kwargs",
#     (
#         pytest.param(None, None, {}, id="no-units-set"),
#         pytest.param("month", None, {}, id="x-units-set"),
#         pytest.param(None, "t / yr", {}, id="y-units-set"),
#         pytest.param("s", "Gt / yr", {}, id="x-and-y-units-set"),
#         pytest.param(
#             "yr", "Gt / yr", dict(alpha=0.7, s=130), id="x-and-y-units-set-kwargs"
#         ),
#     ),
# )
# def test_plot(x_units, y_units, plot_kwargs, image_regression, tmp_path):
#     import matplotlib
#
#     # ensure matplotlib does not use a GUI backend (such as Tk)
#     matplotlib.use("Agg")
#
#     import matplotlib.pyplot as plt
#     import matplotlib.units
#
#     # Setup matplotlib to use units
#     UR.setup_matplotlib(enable=True)
#
#     fig, ax = plt.subplots()
#
#     mt_month = TimeseriesContinuous(
#         name="Mt per month",
#         time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
#         values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Mt / month") * 100.0),
#     )
#
#     gt_yr = TimeseriesContinuous(
#         name="Gt per year",
#         time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
#         values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Gt / yr")),
#     )
#
#     mt = TimeseriesContinuous(
#         name="Mt",
#         time_axis=TimeAxis(Q([2020.0, 2021.0, 2022.0], "yr")),
#         values_at_bounds=ValuesAtBounds(Q([10.0, 10.5, 11.0], "Mt")),
#     )
#
#     if x_units is not None:
#         ax.set_xlabel(x_units)
#         ax.xaxis.set_units(x_units)
#
#     if y_units is not None:
#         ax.set_ylabel(y_units)
#         ax.yaxis.set_units(y_units)
#
#     # Even though timeseries are in different units,
#     # use of pint with matplotib will ensure sensible units on plot.
#     mt_month.plot(ax=ax, **plot_kwargs)
#     gt_yr.plot(ax=ax, **plot_kwargs)
#
#     # Trying to plot something with incompatible units will raise.
#     with pytest.raises(matplotlib.units.ConversionError):
#         mt.plot(ax=ax, **plot_kwargs)
#
#     ax.legend()
#
#     fig.tight_layout()
#
#     out_file = tmp_path / "fig.png"
#     fig.savefig(out_file)
#
#     image_regression.check(out_file.read_bytes(), diff_threshold=0.01)
#
#     # Ensure we tear down
#     UR.setup_matplotlib(enable=False)
#
#
# @pytest.mark.parametrize(
#     "plot_kwargs, legend, expectation",
#     (
#         pytest.param(
#             {},
#             False,
#             pytest.warns(
#                 UserWarning,
#                 match=re.escape(
#                     "The units of `self.values_at_bounds.values` "
#                     "are not registered with matplotlib. "
#                     "The magnitude will be plotted without any consideration of units. "  # noqa: E501
#                     "For docs on how to set up unit-aware plotting, see "
#                     "[the stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html) "  # noqa: E501
#                     "(at the time of writing, the latest version's docs were "
#                     "[v0.24.4](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))."
#                 ),
#             ),
#             id="defaults",
#         ),
#         pytest.param({}, True, pytest.warns(UserWarning), id="legend"),
#         pytest.param(
#             dict(warn_if_plotting_magnitudes=False),
#             True,
#             does_not_raise(),
#             id="no-warning",
#         ),
#         pytest.param(
#             dict(label="custom"),
#             True,
#             pytest.warns(UserWarning),
#             id="label-overwriting",
#         ),
#         pytest.param(
#             dict(marker="x", color="tab:green", label="demo"),
#             True,
#             pytest.warns(UserWarning),
#             id="kwargs-passing",
#         ),
#     ),
# )
# def test_plot_matplotlib_no_units(
#     plot_kwargs, legend, expectation, image_regression, tmp_path
# ):
#     import matplotlib
#
#     # ensure matplotlib does not use a GUI backend (such as Tk)
#     matplotlib.use("Agg")
#
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#
#     ts = TimeseriesContinuous(
#         name="test_plot",
#         time_axis=TimeAxis(
#             Q(
#                 [
#                     1750.0,
#                     1950.0,
#                     1975.0,
#                     2000.0,
#                     2010.0,
#                     2020.0,
#                     2030.0,
#                     2050.0,
#                     2100.0,
#                     2200.0,
#                     2300.0,
#                 ],
#                 "yr",
#             )
#         ),
#         values_at_bounds=ValuesAtBounds(
#             Q(
#                 [0.0, 2.3, 6.4, 10.0, 11.0, 12.3, 10.2, 0.0, -5.0, -2.0, 0.3],
#                 "Gt / yr",
#             )
#         ),
#     )
#
#     with expectation:
#         ts.plot(ax=ax, **plot_kwargs)
#
#     if legend:
#         ax.legend()
#
#     out_file = tmp_path / "fig.png"
#     fig.savefig(out_file)
#
#     image_regression.check(out_file.read_bytes(), diff_threshold=0.01)
#
#
# @pytest.mark.parametrize(
#     "sys_modules_patch, expectation",
#     (
#         pytest.param({}, does_not_raise(), id="matplotlib_available"),
#         pytest.param(
#             {"matplotlib": None},
#             pytest.raises(
#                 MissingOptionalDependencyError,
#                 match="`TimeseriesContinuous.plot` requires matplotlib to be installed",  # noqa: E501
#             ),
#             id="matplotlib_not_available",
#         ),
#     ),
# )
# def test_plot_ax_creation(sys_modules_patch, expectation):
#     ts = TimeseriesContinuous(
#         name="basic",
#         time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
#         values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
#     )
#     with patch.dict(sys.modules, sys_modules_patch):
#         with expectation:
#             ts.plot(warn_if_plotting_magnitudes=False)
#
#
# @pytest.mark.parametrize(
#     "sys_modules_patch, expectation",
#     (
#         pytest.param({}, does_not_raise(), id="matplotlib_available"),
#         pytest.param(
#             {"matplotlib": None},
#             pytest.warns(
#                 UserWarning,
#                 match=re.escape(
#                     "Could not import `matplotlib.units` "
#                     "to set up unit-aware plotting. "
#                     "We will simply try plotting magnitudes instead."
#                 ),
#             ),
#             id="matplotlib_not_available",
#         ),
#     ),
# )
# def test_plot_no_matplotlib_units(sys_modules_patch, expectation):
#     import matplotlib.pyplot as plt
#
#     _, ax = plt.subplots()
#
#     ts = TimeseriesContinuous(
#         name="basic",
#         time_axis=TimeAxis(Q([1.0, 2.0, 3.0], "yr")),
#         values_at_bounds=ValuesAtBounds(Q([10.0, 20.0, 5.0], "kg")),
#     )
#     with patch.dict(sys.modules, sys_modules_patch):
#         with expectation:
#             ts.plot(ax=ax)
