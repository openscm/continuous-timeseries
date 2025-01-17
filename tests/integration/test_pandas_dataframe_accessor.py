"""
Test our pandas DataFrame accessors

In other words, test the `pd.DataFrame.ct` namespace.
This may need to be split into multiple files in future.
"""

from __future__ import annotations

import pint
import pytest

import continuous_timeseries as ct
from continuous_timeseries.pandas_accessors import register_pandas_accessor

pd = pytest.importorskip("pandas")

UR = pint.get_application_registry()
Q = UR.Quantity


def test_to_timeseries():
    if not hasattr(pd.DataFrame, "ct"):
        register_pandas_accessor()

    x = Q([2010, 2015, 2025], "yr")
    y_ms = [
        [1.0, 2.0, 3.0],
        [-1.5, -0.5, 0.5],
    ]
    idx = pd.MultiIndex.from_tuples(
        (
            ("name_1", "Mt  /yr"),
            ("name_2", "Gt"),
        ),
        # units not unit to follow pint conventions
        names=["name", "units"],
    )

    df = pd.DataFrame(
        y_ms,
        columns=x.m,
        index=idx,
    )

    res = df.ct.to_timeseries(
        time_units=x.u,
        interpolation=ct.InterpolationOption.PiecewiseConstantPreviousLeftClosed,
    )

    # Check results same as just doing a dumb loop here

    # Then test going back to df i.e. round tripping
    # - also with different length time axes
    # - different units
    # Test plotting
    # Test ops pass through
    # - use a series accessor for this
    # Test on 600 x 2000 x 10 timeseries (length ~450 values each),
    # also helps check parallelisation.
    # This would be a 50GB array so maybe a stupid use case already...

    assert False

    # Would be nice to deregister too so other tests can check.
    # De-registering does not seem to be that easy though...
