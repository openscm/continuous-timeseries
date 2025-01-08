"""
Integration tests of `continuous_timeseries.timeseries_continuous`
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
from attrs import define

from continuous_timeseries.exceptions import (
    ExtrapolationNotAllowedError,
    MissingOptionalDependencyError,
)
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_continuous import (
    ContinuousFunctionScipyPPoly,
    TimeseriesContinuous,
)

scipy = pytest.importorskip("scipy")

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "continuous_function_scipy_ppoly, exp_re",
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
def test_repr_continuous_function_scipy_ppoly(continuous_function_scipy_ppoly, exp_re):
    repr_value = repr(continuous_function_scipy_ppoly)

    assert exp_re.fullmatch(repr_value)


@pytest.mark.parametrize(
    "continuous_function_scipy_ppoly, exp",
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
def test_str_continuous_function_scipy_ppoly(continuous_function_scipy_ppoly, exp):
    str_value = str(continuous_function_scipy_ppoly)

    assert str_value == exp


@pytest.mark.parametrize(
    "continuous_function_scipy_ppoly, exp",
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
            marks=pytest.mark.xfail(
                condition=not (sys.version_info >= (3, 10)),
                reason="shape info only in Python>=3.10",
            ),
            id="heaps_of_windows",
        ),
    ),
)
def test_pretty_continuous_function_scipy_ppoly(continuous_function_scipy_ppoly, exp):
    pytest.importorskip("IPython")

    from IPython.lib.pretty import pretty

    pretty_value = pretty(continuous_function_scipy_ppoly)

    assert pretty_value == exp


@pytest.mark.parametrize(
    "continuous_function_scipy_ppoly",
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
def test_html_continuous_function_scipy_ppoly(
    continuous_function_scipy_ppoly, file_regression
):
    file_regression.check(
        f"{continuous_function_scipy_ppoly._repr_html_()}\n",
        extension=".html",
    )


@pytest.mark.parametrize(
    "continuous_function_scipy_ppoly, order_exp, order_str_exp",
    (
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
            ),
            0,
            "0th",
            id="piecewise_constant",
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(x=[1, 10, 20], c=[[1.0, 2.0], [10, 12]])
            ),
            1,
            "1st",
            id="piecewise_linear",
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=[1, 10, 20], c=[[1.2, -9.5], [1.0, 2.0], [10, 12]]
                )
            ),
            2,
            "2nd",
            id="piecewise_quadratic",
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=[1, 10, 20], c=[[0.02, 0.03], [1.2, -9.5], [1.0, 2.0], [10, 12]]
                )
            ),
            3,
            "3rd",
            id="piecewise_cubic",
        ),
        pytest.param(
            ContinuousFunctionScipyPPoly(
                scipy.interpolate.PPoly(
                    x=[1, 10, 20],
                    c=[[0.0, 0.001], [0.02, 0.03], [1.2, -9.5], [1.0, 2.0], [10, 12]],
                )
            ),
            4,
            "4th",
            id="piecewise_quartic",
        ),
    ),
)
def test_order_continuous_function_scipy_ppoly(
    continuous_function_scipy_ppoly, order_exp, order_str_exp
):
    assert continuous_function_scipy_ppoly.order == order_exp
    assert continuous_function_scipy_ppoly.order_str == order_str_exp


formatting_check_cases = pytest.mark.parametrize(
    "ts",
    (
        pytest.param(
            TimeseriesContinuous(
                name="piecewise_constant",
                time_units=UR.Unit("yr"),
                values_units=UR.Unit("Gt"),
                function=ContinuousFunctionScipyPPoly(
                    scipy.interpolate.PPoly(x=[1, 10, 20], c=[[10, 12]])
                ),
                domain=(Q(1, "yr"), Q(20, "yr")),
            ),
            id="piecewise_constant",
        ),
        pytest.param(
            TimeseriesContinuous(
                name="piecewise_linear",
                time_units=UR.Unit("yr"),
                values_units=UR.Unit("Gt"),
                function=ContinuousFunctionScipyPPoly(
                    scipy.interpolate.PPoly(x=[1, 10, 20], c=[[-1.2, 2.3], [10, 12]])
                ),
                domain=(Q(1, "yr"), Q(20, "yr")),
            ),
            id="piecewise_linear",
        ),
        pytest.param(
            TimeseriesContinuous(
                name="piecewise_linear_heaps_of_windows",
                time_units=UR.Unit("yr"),
                values_units=UR.Unit("Gt"),
                function=ContinuousFunctionScipyPPoly(
                    scipy.interpolate.PPoly(
                        x=np.arange(10001), c=np.arange(20000).reshape(2, 10000)
                    )
                ),
                domain=(Q(0, "yr"), Q(1000, "yr")),
            ),
            id="piecewise_linear_heaps_of_windows",
        ),
    ),
)


@formatting_check_cases
def test_repr(ts, file_regression):
    exp = (
        "TimeseriesContinuous("
        f"name={ts.name!r}, "
        f"time_units={ts.time_units!r}, "
        f"values_units={ts.values_units!r}, "
        f"function={ts.function!r}, "
        f"domain={ts.domain!r}"
        ")"
    )

    assert repr(ts) == exp

    # Avoid changing addresses causing issues
    file_regression_value = re.sub("at .*>", "at address>", repr(ts))

    file_regression.check(
        f"{file_regression_value}\n",
        extension=".txt",
    )


@formatting_check_cases
def test_str(ts, file_regression):
    exp = (
        "TimeseriesContinuous("
        f"name={ts.name}, "
        f"time_units={ts.time_units}, "
        f"values_units={ts.values_units}, "
        f"function={ts.function}, "
        f"domain={ts.domain}"
        ")"
    )

    assert str(ts) == exp

    file_regression.check(
        f"{ts}\n",
        extension=".txt",
    )


@pytest.mark.xfail(
    condition=not (sys.version_info >= (3, 10)),
    reason="shape info only in Python>=3.10",
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


@define
class OperationsTestCase:
    """A test case for operations with `TimeseriesContinuous`"""

    ts: TimeseriesContinuous
    """Timeseries to use for the tests"""

    time_interp: UR.Quantity
    """Times to use for checking interpolation"""

    exp_interp: UR.Quantity
    """Expected values of interpolation at `time_interp`"""

    time_extrap: UR.Quantity
    """Times to use for checking extrapolation"""

    exp_extrap: UR.Quantity
    """Expected values of extrapolation at `time_extrap`"""

    time_integral_check: UR.Quantity
    """Times to use for checking the values of the integral"""

    exp_integral_values_excl_integration_constant: UR.Quantity
    """
    Expected values of the derivate at `time_integral_check`

    This excludes the integration constant
    (i.e. assumes the integration constant is zero).
    """

    time_derivative_check: UR.Quantity
    """Times to use for checking the values of the derivative"""

    exp_derivative_values: UR.Quantity
    """Expected values of the derivate at `time_derivative_check`"""


operations_test_cases = pytest.mark.parametrize(
    "operations_test_case",
    (
        pytest.param(
            OperationsTestCase(
                ts=TimeseriesContinuous(
                    name="piecewise_constant",
                    time_units=UR.Unit("yr"),
                    values_units=UR.Unit("Gt"),
                    function=ContinuousFunctionScipyPPoly(
                        scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[2.5]])
                    ),
                    domain=(Q(1.0, "yr"), Q(2.0, "yr")),
                ),
                time_interp=Q([1.25, 1.5, 1.75], "yr"),
                exp_interp=Q([2.5, 2.5, 2.5], "Gt"),
                time_extrap=Q([0.0, 1.0, 2.0, 3.0], "yr"),
                exp_extrap=Q([2.5, 2.5, 2.5, 2.5], "Gt"),
                time_integral_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_integral_values_excl_integration_constant=Q(
                    [
                        0.0,
                        2.5 / 2,
                        2.5,
                    ],
                    "Gt yr",
                ),
                time_derivative_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_derivative_values=Q([0.0, 0.0, 0.0], "Gt / yr"),
            ),
            id="basic_constant",
        ),
        pytest.param(
            OperationsTestCase(
                ts=TimeseriesContinuous(
                    name="piecewise_linear",
                    time_units=UR.Unit("yr"),
                    values_units=UR.Unit("Gt"),
                    function=ContinuousFunctionScipyPPoly(
                        scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [2.5]])
                    ),
                    domain=(Q(1.0, "yr"), Q(2.0, "yr")),
                ),
                time_interp=Q([1.25, 1.5, 1.75], "yr"),
                exp_interp=Q([2.75, 3.0, 3.25], "Gt"),
                time_extrap=Q([0.0, 1.0, 2.0, 3.0], "yr"),
                exp_extrap=Q([1.5, 2.5, 3.5, 4.5], "Gt"),
                time_integral_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_integral_values_excl_integration_constant=Q(
                    [
                        0.0,
                        (1 / 2) ** 2 / 2 + 2.5 / 2,
                        1 / 2 + 2.5,
                    ],
                    "Gt yr",
                ),
                time_derivative_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_derivative_values=Q([1.0, 1.0, 1.0], "Gt / yr"),
            ),
            id="basic_linear",
        ),
        pytest.param(
            OperationsTestCase(
                ts=TimeseriesContinuous(
                    name="piecewise_quadratic",
                    time_units=UR.Unit("yr"),
                    values_units=UR.Unit("Gt"),
                    function=ContinuousFunctionScipyPPoly(
                        scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [1.0], [0.0]])
                    ),
                    domain=(Q(1.0, "yr"), Q(2.0, "yr")),
                ),
                time_interp=Q([1.25, 1.5, 1.75], "yr"),
                exp_interp=Q([1 / 16 + 1 / 4, 1 / 4 + 1 / 2, 9 / 16 + 3 / 4], "Gt"),
                time_extrap=Q([0.0, 0.5, 1.0, 2.0, 3.0], "yr"),
                exp_extrap=Q([0.0, 1 / 4 - 1 / 2, 0.0, 2.0, 6.0], "Gt"),
                time_integral_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_integral_values_excl_integration_constant=Q(
                    [
                        0.0,
                        (1 / 2) ** 3 / 3 + (1 / 2) ** 2 / 2,
                        1 / 3 + 1 / 2,
                    ],
                    "Gt yr",
                ),
                time_derivative_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_derivative_values=Q([1.0, 2.0, 3.0], "Gt / yr"),
            ),
            id="basic_quadratic",
        ),
        pytest.param(
            OperationsTestCase(
                ts=TimeseriesContinuous(
                    name="time_axis_is_time_axis",
                    time_units=UR.Unit("yr"),
                    values_units=UR.Unit("Gt"),
                    function=ContinuousFunctionScipyPPoly(
                        scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [2.5]])
                    ),
                    domain=(Q(1.0, "yr"), Q(2.0, "yr")),
                ),
                time_interp=TimeAxis(Q([1.25, 1.5, 1.75], "yr")),
                exp_interp=Q([2.75, 3.0, 3.25], "Gt"),
                time_extrap=TimeAxis(Q([0.0, 1.0, 2.0, 3.0], "yr")),
                exp_extrap=Q([1.5, 2.5, 3.5, 4.5], "Gt"),
                time_integral_check=Q([1.0, 1.5, 2.0], "yr"),
                exp_integral_values_excl_integration_constant=Q(
                    [
                        0.0,
                        (1 / 2) ** 2 / 2 + 2.5 / 2,
                        1 / 2 + 2.5,
                    ],
                    "Gt yr",
                ),
                time_derivative_check=TimeAxis(Q([1.0, 1.5, 2.0], "yr")),
                exp_derivative_values=Q([1.0, 1.0, 1.0], "Gt / yr"),
            ),
            id="time_axis_is_time_axis",
        ),
        pytest.param(
            OperationsTestCase(
                ts=TimeseriesContinuous(
                    name="linear_using_unit_conversion",
                    time_units=UR.Unit("yr"),
                    values_units=UR.Unit("Gt"),
                    function=ContinuousFunctionScipyPPoly(
                        scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [2.5]])
                    ),
                    domain=(Q(1.0, "yr"), Q(2.0, "yr")),
                ),
                time_interp=Q([15, 18, 21], "month"),
                exp_interp=Q([2750, 3000, 3250], "Mt"),
                time_extrap=Q([0, 12, 24, 36], "month"),
                exp_extrap=Q([1500, 2500, 3500, 4500], "Mt"),
                time_integral_check=Q([12, 18, 24], "month"),
                exp_integral_values_excl_integration_constant=Q(
                    [
                        0.0,
                        1000.0 * 12.0 * ((1 / 2) ** 2 / 2 + 2.5 / 2),
                        1000.0 * 12.0 * (1 / 2 + 2.5),
                    ],
                    "Mt month",
                ),
                time_derivative_check=Q([12, 18, 24], "month"),
                exp_derivative_values=Q(
                    [
                        1000.0 / 12,
                        1000.0 / 12,
                        1000.0 / 12,
                    ],
                    "Mt / month",
                ),
            ),
            id="linear_using_unit_conversion",
        ),
    ),
)


@operations_test_cases
def test_interpolate(operations_test_case):
    pint.testing.assert_allclose(
        operations_test_case.ts.interpolate(operations_test_case.time_interp),
        operations_test_case.exp_interp,
        rtol=1e-10,
    )


@operations_test_cases
def test_extrapolate_not_allowed_raises(operations_test_case):
    with pytest.raises(ExtrapolationNotAllowedError):
        operations_test_case.ts.interpolate(operations_test_case.time_extrap)


@operations_test_cases
def test_extrapolate(operations_test_case):
    pint.testing.assert_allclose(
        operations_test_case.ts.interpolate(
            operations_test_case.time_extrap, allow_extrapolation=True
        ),
        operations_test_case.exp_extrap,
        rtol=1e-10,
    )


@operations_test_cases
@pytest.mark.parametrize(
    "integration_constant",
    (
        Q(0, "Gt yr"),
        Q(1.0, "Gt yr"),
    ),
)
@pytest.mark.parametrize(
    "integrate_kwargs", ({}, dict(name_res=None), dict(name_res="name_overwritten"))
)
def test_integration(operations_test_case, integration_constant, integrate_kwargs):
    integral = operations_test_case.ts.integrate(
        integration_constant=integration_constant, **integrate_kwargs
    )

    pint.testing.assert_allclose(
        integral.interpolate(operations_test_case.time_integral_check),
        operations_test_case.exp_integral_values_excl_integration_constant
        + integration_constant,
        rtol=1e-10,
    )

    if (
        integrate_kwargs
        and "name_res" in integrate_kwargs
        and integrate_kwargs["name_res"] is not None
    ):
        assert integral.name == integrate_kwargs["name_res"]
    else:
        assert integral.name == f"{operations_test_case.ts.name}_integral"


@operations_test_cases
@pytest.mark.parametrize(
    "differentiate_kwargs", ({}, dict(name_res=None), dict(name_res="name_overwritten"))
)
def test_differentiate(operations_test_case, differentiate_kwargs):
    derivative = operations_test_case.ts.differentiate(**differentiate_kwargs)

    pint.testing.assert_allclose(
        derivative.interpolate(operations_test_case.time_derivative_check),
        operations_test_case.exp_derivative_values,
        rtol=1e-10,
    )

    if (
        differentiate_kwargs
        and "name_res" in differentiate_kwargs
        and differentiate_kwargs["name_res"] is not None
    ):
        assert derivative.name == differentiate_kwargs["name_res"]
    else:
        assert derivative.name == f"{operations_test_case.ts.name}_derivative"


@pytest.mark.parametrize(
    "x_units, y_units, plot_kwargs, legend",
    (
        pytest.param(None, None, {}, False, id="no-units-set"),
        pytest.param("month", None, {}, False, id="x-units-set"),
        pytest.param(None, "t", {}, False, id="y-units-set"),
        pytest.param("s", "Gt", {}, False, id="x-and-y-units-set"),
        pytest.param(None, None, {}, True, id="default-labels"),
        pytest.param(
            None, None, dict(label="overwritten"), True, id="overwrite-labels"
        ),
        pytest.param(
            "yr",
            "Gt",
            dict(alpha=0.7, linewidth=2),
            False,
            id="x-and-y-units-set-kwargs",
        ),
        pytest.param(
            None,
            None,
            dict(res_increase=2, label="res_increase=2"),
            True,
            id="res-increase",
        ),
    ),
)
@pytest.mark.parametrize(
    "time_axis_plot",
    (
        pytest.param(Q([1.0, 2.0], "yr"), id="basic"),
        pytest.param(TimeAxis(Q([1.0, 2.0], "yr")), id="TimeAxis"),
        pytest.param(Q([1.25, 1.75], "yr"), id="within_domain"),
    ),
)
def test_plot(  # noqa: PLR0913
    x_units, y_units, plot_kwargs, legend, time_axis_plot, image_regression, tmp_path
):
    import matplotlib

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.units

    # Setup matplotlib to use units
    UR.setup_matplotlib(enable=True)

    fig, ax = plt.subplots()

    gt = TimeseriesContinuous(
        name="gt_quadratic",
        time_units=UR.Unit("yr"),
        values_units=UR.Unit("Gt"),
        function=ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [1.0], [0.0]])
        ),
        domain=(Q(1.0, "yr"), Q(2.0, "yr")),
    )
    mt = TimeseriesContinuous(
        name="mt_linear",
        time_units=UR.Unit("yr"),
        values_units=UR.Unit("Mt"),
        function=ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1000.0], [3.0]])
        ),
        domain=(Q(1.0, "yr"), Q(2.0, "yr")),
    )

    gt_per_year = TimeseriesContinuous(
        name="gt_quadratic_per_year",
        time_units=UR.Unit("yr"),
        values_units=UR.Unit("Gt / yr"),
        function=ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [1.0], [0.0]])
        ),
        domain=(Q(1.0, "yr"), Q(2.0, "yr")),
    )

    if x_units is not None:
        ax.set_xlabel(x_units)
        ax.xaxis.set_units(UR.Unit(x_units))

    if y_units is not None:
        ax.set_ylabel(y_units)
        ax.yaxis.set_units(UR.Unit(y_units))

    # Even though timeseries are in different units,
    # use of pint with matplotib will ensure sensible units on plot.
    mt.plot(ax=ax, time_axis=time_axis_plot, **plot_kwargs)
    gt.plot(ax=ax, time_axis=time_axis_plot, **plot_kwargs)

    # Trying to plot something with incompatible units will raise.
    with pytest.raises(matplotlib.units.ConversionError):
        gt_per_year.plot(ax=ax, time_axis=time_axis_plot, **plot_kwargs)

    if legend:
        ax.legend()

    fig.tight_layout()

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)

    # Ensure we tear down
    UR.setup_matplotlib(enable=False)
    plt.close()


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
    import matplotlib

    # ensure matplotlib does not use a GUI backend (such as Tk)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ts = TimeseriesContinuous(
        name="piecewise_quadratic",
        time_units=UR.Unit("yr"),
        values_units=UR.Unit("Gt"),
        function=ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [1.0], [0.0]])
        ),
        domain=(Q(1.0, "yr"), Q(2.0, "yr")),
    )

    time_axis_plot = Q([1.0, 2.0], "yr")
    with expectation:
        ts.plot(ax=ax, time_axis=time_axis_plot, **plot_kwargs)

    out_file = tmp_path / "fig.png"
    fig.savefig(out_file)

    plt.close()

    image_regression.check(out_file.read_bytes(), diff_threshold=0.01)


@pytest.mark.parametrize(
    "sys_modules_patch, expectation",
    (
        pytest.param({}, does_not_raise(), id="matplotlib_available"),
        pytest.param(
            {"matplotlib": None},
            pytest.raises(
                MissingOptionalDependencyError,
                match="`TimeseriesContinuous.plot` requires matplotlib to be installed",
            ),
            id="matplotlib_not_available",
        ),
    ),
)
def test_plot_ax_creation(sys_modules_patch, expectation):
    ts = TimeseriesContinuous(
        name="piecewise_quadratic",
        time_units=UR.Unit("yr"),
        values_units=UR.Unit("Gt"),
        function=ContinuousFunctionScipyPPoly(
            scipy.interpolate.PPoly(x=[1.0, 2.0], c=[[1.0], [1.0], [0.0]])
        ),
        domain=(Q(1.0, "yr"), Q(2.0, "yr")),
    )
    with patch.dict(sys.modules, sys_modules_patch):
        with expectation:
            ts.plot(
                time_axis=Q(ts.function.ppoly.x, ts.time_units),
                warn_if_plotting_magnitudes=False,
            )
