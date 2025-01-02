"""
Representation of continuous timeseries.
"""

import importlib.metadata

from continuous_timeseries.discrete_to_continuous import (
    InterpolationOption,
    discrete_to_continuous,
)
from continuous_timeseries.time_axis import TimeAxis
from continuous_timeseries.timeseries_continuous import TimeseriesContinuous
from continuous_timeseries.timeseries_discrete import TimeseriesDiscrete
from continuous_timeseries.values_at_bounds import ValuesAtBounds

__version__ = importlib.metadata.version("continuous_timeseries")

__all__ = [
    "InterpolationOption",
    "TimeAxis",
    "TimeseriesContinuous",
    "TimeseriesDiscrete",
    "ValuesAtBounds",
    "discrete_to_continuous",
]


Timeseries = None
"""Placeholder so that docs build before we implement this"""
