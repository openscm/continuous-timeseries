"""
Representation of continuous timeseries.
"""

import importlib.metadata

from continuous_timeseries.time_axis import TimeAxis

__version__ = importlib.metadata.version("continuous_timeseries")

__all__ = ["TimeAxis"]

InterpolationOption = None
"""Placeholder so that docs build before we implement this"""

Timeseries = None
"""Placeholder so that docs build before we implement this"""

TimeseriesContinuous = None
"""Placeholder so that docs build before we implement this"""

TimeseriesDiscrete = None
"""Placeholder so that docs build before we implement this"""
