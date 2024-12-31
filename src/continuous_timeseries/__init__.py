"""
Representation of continuous timeseries.
"""

import importlib.metadata

from .timeseries import Timeseries

__all__ = [
    "Timeseries",
]

__version__ = importlib.metadata.version("continuous_timeseries")
