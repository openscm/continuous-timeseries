"""
Definition of a our interpolation options (represented by [`InterpolationOption`][(m)]).

At the moment,
only the piecewise constant options support the left-open/left-closed concept.
In theory, this could also apply to the other interpolation options.
However, this only matters if we want to support discontinuous values
at the boundaries for these interpolation choices.
So, while we could support this, we currently don't
because the use case for discontinuous values
in combination with linear or higher order interpolation isn't clear.
"""

from __future__ import annotations

from enum import IntEnum, unique


@unique
class InterpolationOption(IntEnum):
    """
    Interpolation options
    """

    Linear = 1
    """Linear interpolation"""

    Quadratic = 2
    """Quadratic interpolation"""

    Cubic = 3
    """Cubic interpolation"""

    Quartic = 4
    """Quartic interpolation"""

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

    ```
    y(4):                                    ixxxxxxxxxxxxxxxxxxxxxxxxxx
    y(3):                        ixxxxxxxxxxxo
    y(2):            ixxxxxxxxxxxo
    y(1): xxxxxxxxxxxo
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    ```
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

    ```
    y(4):                                    oxxxxxxxxxxxxxxxxxxxxxxxxxx
    y(3):                        oxxxxxxxxxxxi
    y(2):            oxxxxxxxxxxxi
    y(1): xxxxxxxxxxxi
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    ```
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

    ```
    y(4):                                                ixxxxxxxxxxxxxx
    y(3):                                    ixxxxxxxxxxxo
    y(2):                        ixxxxxxxxxxxo
    y(1): xxxxxxxxxxxxxxxxxxxxxxxo
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    ```
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

    ```
    y(4):                                                oxxxxxxxxxxxxxx
    y(3):                                    oxxxxxxxxxxxi
    y(2):                        oxxxxxxxxxxxi
    y(1): xxxxxxxxxxxxxxxxxxxxxxxi
          -----------|-----------|-----------|-----------|--------------
                  time(1)     time(2)     time(3)     time(4)
    ```
    """
