"""
Definition of the interpolation options we support
"""

from __future__ import annotations

from enum import UNIQUE, IntEnum, verify


@verify(UNIQUE)
class InterpolationOption(IntEnum):
    """
    Interpolation options
    """

    NotSpecified = 0
    """No handling has been specified"""

    Linear = 1
    """Linear interpolation is assumed between points"""

    Quadratic = 2
    """Quadratic interpolation is assumed between points"""

    Cubic = 3
    """Cubic interpolation is assumed between points"""

    PiecewiseConstantNextLeftClosed = 10
    """
    Piecewise constant 'next' interpolation, each interval is closed on the left

    In other words,
    between t(i) and t(i + 1), the value is equal to y(i + 1).
    At t(i), the value is equal to y(i + 1).

    If helpful, we have drawn a picture of how this works below.
    Symbols:

    - time: y-value selected for this time-value
    - i: closed (i.e. inclusive) boundary
    - o: open (i.e. exclusive) boundary

    y(4):                                    ixxxxxxxxxxxxxxxxxxxxxxxxxx
    y(3):                        ixxxxxxxxxxxo
    y(2):            ixxxxxxxxxxxo
    y(1): xxxxxxxxxxxo
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    """

    PiecewiseConstantNextLeftOpen = 11
    """
    Piecewise constant 'next' interpolation, each interval is open on the left

    In other words,
    between t(i) and t(i + 1), the value is equal to y(i + 1).
    At t(i), the value is equal to y(i).

    If helpful, we have drawn a picture of how this works below.
    Symbols:

    - time: y-value selected for this time-value
    - i: closed (i.e. inclusive) boundary
    - o: open (i.e. exclusive) boundary

    y(4):                                    oxxxxxxxxxxxxxxxxxxxxxxxxxx
    y(3):                        oxxxxxxxxxxxi
    y(2):            oxxxxxxxxxxxi
    y(1): xxxxxxxxxxxi
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    """

    PiecewiseConstantPreviousLeftClosed = 12
    """
    Piecewise constant 'previous' interpolation, each interval is closed on the left

    In other words,
    between t(i) and t(i + 1), the value is equal to y(i).
    At t(i + 1), the value is equal to y(i + 1).

    If helpful, we have drawn a picture of how this works below.
    Symbols:

    - time: y-value selected for this time-value
    - i: closed (i.e. inclusive) boundary
    - o: open (i.e. exclusive) boundary

    y(4):                                                ixxxxxxxxxxxxxx
    y(3):                                    ixxxxxxxxxxxo
    y(2):                        ixxxxxxxxxxxo
    y(1): xxxxxxxxxxxxxxxxxxxxxxxo
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    """

    PiecewiseConstantPreviousLeftOpen = 13
    """
    Piecewise constant 'previous' interpolation, each interval is open on the left

    In other words,
    between t(i) and t(i + 1), the value is equal to y(i).
    At t(i + 1), the value is equal to y(i).

    If helpful, we have drawn a picture of how this works below.
    Symbols:

    - time: y-value selected for this time-value
    - i: closed (i.e. inclusive) boundary
    - o: open (i.e. exclusive) boundary

    y(4):                                                oxxxxxxxxxxxxxx
    y(3):                                    oxxxxxxxxxxxi
    y(2):                        oxxxxxxxxxxxi
    y(1): xxxxxxxxxxxxxxxxxxxxxxxi
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    """
