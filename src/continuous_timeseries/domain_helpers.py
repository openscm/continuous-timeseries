"""
Support for our domain handling
"""

from __future__ import annotations

import numpy as np

from continuous_timeseries.typing import PINT_NUMPY_ARRAY, PINT_SCALAR


def check_no_times_outside_domain(
    times: PINT_NUMPY_ARRAY,
    domain: tuple[PINT_SCALAR, PINT_SCALAR],
) -> None:
    """
    Check that no times are outside the supported domain

    Parameters
    ----------
    times
        Times to check

    domain
        Supported domain

    Raises
    ------
    ValueError
        There are values in `time` that are outside the supported domain.

    AssertionError
        `len(domain) != 2` or `domain[1] <= domain[0]`.
    """
    expected_domain_length = 2
    if len(domain) != expected_domain_length:
        raise AssertionError(len(domain))

    if domain[1] <= domain[0]:
        msg = f"domain[1] must be greater than domain[0]. Received {domain=}."

        raise AssertionError(msg)

    outside_domain = np.hstack(
        [
            times[np.where(times < domain[0])],
            times[np.where(times > domain[1])],
        ]
    )

    if outside_domain.size >= 1:
        msg = (
            f"The {domain=}. "
            "There are time values that are outside this domain. "
            f"{outside_domain=}. "
        )
        raise ValueError(msg)
