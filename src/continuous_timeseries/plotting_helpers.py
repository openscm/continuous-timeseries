"""
Support for plotting
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from continuous_timeseries.typing import PINT_NUMPY_ARRAY


def get_plot_vals(
    pint_q: PINT_NUMPY_ARRAY, desc: str, warn_if_magnitudes: bool
) -> PINT_NUMPY_ARRAY | npt.NDArray[np.number[Any]]:
    """
    Get values to plot

    This function is a helper that ensures that we use pint objects
    when matplotlib knows about them but plain numpy arrays otherwise.

    Parameters
    ----------
    pint_q
        Pint quantity from which to get the plot values.

    desc
        Descripion of `pint_q`, used in creating messages.

    warn_if_magnitudes
        Should a warning be raised if magnitudes will be returned?

        This helps alert users
        that they haven't set matplotlib up to be unit-aware with pint.

    Returns
    -------
    :
        Values to plot

    Warns
    -----
    UserWarning
        Magnitudes will be returned and `warn_if_plotting_magnitudes` is `True`.

        Also warns if `matplotlib.units` could not be imported for some reason
        (e.g. the user is using an old version of matplotlib).
        In this case, magnitudes are simply returned.
    """
    try:
        import matplotlib.units

        units_registered_with_matplotlib = type(pint_q) in matplotlib.units.registry

    except ImportError:
        msg = (
            "Could not import `matplotlib.units` "
            "to set up unit-aware plotting. "
            "We will simply try plotting magnitudes instead."
        )
        warnings.warn(msg, stacklevel=3)

        return pint_q.m

    if units_registered_with_matplotlib:
        return pint_q

    if warn_if_magnitudes:
        msg = (
            f"The units of `{desc}` are not registered with matplotlib. "
            "The magnitude will be plotted "
            "without any consideration of units. "
            "For docs on how to set up unit-aware plotting, see "
            "[the stable docs](https://pint.readthedocs.io/en/stable/user/plotting.html) "  # noqa: E501
            "(at the time of writing, the latest version's docs were "
            "[v0.24.4](https://pint.readthedocs.io/en/0.24.4/user/plotting.html))."
        )
        warnings.warn(msg, stacklevel=3)

    return pint_q.m
