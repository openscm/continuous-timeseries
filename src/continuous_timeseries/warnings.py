"""
Warnings that are used throughout
"""

from __future__ import annotations


class InterpolationUpdateChangedValuesAtBoundsWarning(UserWarning):
    """
    Raised when a user does an incompatible interpolation update

    Put another way, when a user converts the interpolation of a timeseries
    in a way that may have confusing results.
    """

    def __init__(self, message: str) -> None:
        self.message = message
