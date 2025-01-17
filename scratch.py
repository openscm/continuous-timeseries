# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import pint
from attrs import define

import continuous_timeseries as ct
from continuous_timeseries import InterpolationOption, Timeseries

# %%
UR = pint.UnitRegistry()
Q = UR.Quantity

# %%
UR.setup_matplotlib(enable=True)

# %%
x = Q([2009, 2010, 2011, 2012, 2013, 2014], "yr")
ts = Timeseries.from_arrays(
    x=x,
    y=Q([1.0, 1.0, 2.0, 3.0, 1.5, 2.0], "kg"),
    interpolation=InterpolationOption.PiecewiseConstantPreviousLeftOpen,
    name="start",
)
ts.plot()

# %%
x_month = Q(
    [y + m / 12 for y in range(x.m.min(), x.m.max()) for m in range(12)] + [x.m.max()],
    "yr",
)
x_month


# %%
@define
class Mean:
    values: int
    bounds: int


# %%
def mean(ts, out_bounds):
    tmp = 0.0 * (
        ts.timeseries_continuous.values_units * ts.timeseries_continuous.time_units
    )
    integral = ts.interpolate(out_bounds).integrate(tmp)

    integral_discrete = integral.discrete.values_at_bounds.values
    integral_per_window = integral_discrete[1:] - integral_discrete[:-1]

    size_of_windows = out_bounds[1:] - out_bounds[:-1]
    mean_per_window = integral_per_window / size_of_windows

    return Mean(values=mean_per_window, bounds=out_bounds)


# %%
mean(ts, x_month).values

# %%
ts_interp = ts.update_interpolation_integral_preserving(InterpolationOption.Quadratic)

# %%
ts_interp_lin = ts.update_interpolation_integral_preserving(InterpolationOption.Linear)

# %%
ts_interp_lin_mean = mean(ts_interp_lin, out_bounds=x_month)

# %%
mean(ts_interp_lin, x)

# %%
tmp_mean = mean(ts_interp, x_month)

# %%
xc = ts.integrate(Q(0, "kg yr")).discrete.time_axis.bounds[0:]
yc = ts.integrate(Q(0, "kg yr")).discrete.values_at_bounds.values[0:]

# %%
import scipy.interpolate

# %%
cubic_spline = scipy.interpolate.CubicSpline(
    x=xc.m,
    y=yc.m,
    # Scipy's docs on this are very helpful.
    # Here, we are basically saying,
    # make the first derivative at either boundary equal to zero.
    # For our data, this makes sense.
    # For other data, a different choice chould be better
    # (e.g. we often use `bc_type=((1, 0.0), "not-a-knot")`
    # which means, have a first derivative of zero on the left,
    # on the right,
    # just use the same polynomial for the last two time steps).)
    bc_type=((1, 1.0), (2, 0.0)),
)

custom = ct.Timeseries(
    time_axis=ct.TimeAxis(xc),
    timeseries_continuous=ct.TimeseriesContinuouss(
        name="custom_spline",
        time_units=xc.u,
        values_units=yc.u,
        function=ct.timeseries_continuous.ContinuousFunctionScipyPPoly(cubic_spline),
        domain=(xc.min(), xc.max()),
    ),
)
custom

# %%
custom_der = custom.differentiate()

# %%
fig, ax = plt.subplots()

ts.plot(ax=ax)
# ts_interp.plot(ax=ax)
# ts_interp_lin.plot(ax=ax)
custom_der.plot(ax=ax)
# ax.scatter(
#     (tmp_mean.bounds[1:] + tmp_mean.bounds[:-1]) / 2.0,
#     tmp_mean.values,
# )
# ax.scatter(
#     (ts_interp_lin_mean.bounds[1:] + ts_interp_lin_mean.bounds[:-1]) / 2.0,
#     ts_interp_lin_mean.values,
# )

ax.grid()

# %%
mean(ts.update_interpolation_integral_preserving(InterpolationOption.Quadratic), x)

# %%
tmp = (
    ts.update_interpolation_integral_preserving(InterpolationOption.Quadratic)
    .interpolate(x_month)
    .integrate(Q(0, "kg yr"))
    .discrete.values_at_bounds.values
)
finer = (tmp[1:] - tmp[:-1]) / (x_month[1:] - x_month[:-1])
finer

# %%
finer[12:24].sum() / 12

# %%
finer[:12].sum() / 12

# %%
finer[24:36].sum() / 12

# %%
finer[36:48].sum() / 12

# %%
