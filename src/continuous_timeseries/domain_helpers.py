"""
Support for our domain handling
"""

from __future__ import annotations

from typing import Union, overload

import numpy as np

from continuous_timeseries.typing import (
    NP_ARRAY_OF_FLOAT_OR_INT,
    NP_FLOAT_OR_INT,
    PINT_NUMPY_ARRAY,
    PINT_SCALAR,
)


def validate_domain(
    domain: Union[
        tuple[PINT_SCALAR, PINT_SCALAR], tuple[NP_FLOAT_OR_INT, NP_FLOAT_OR_INT]
    ],
) -> None:
    """
    Check that domain values are valid

    Parameters
    ----------
    domain
        Domain to check

    Raises
    ------
    AssertionError
        `len(domain) != 2` or `domain[1] <= domain[0]`.
    """
    expected_domain_length = 2
    if len(domain) != expected_domain_length:
        raise AssertionError(len(domain))

    if domain[1] <= domain[0]:
        msg = f"domain[1] must be greater than domain[0]. Received {domain=}."

        raise AssertionError(msg)


@overload
def check_no_times_outside_domain(
    times: Union[PINT_NUMPY_ARRAY, PINT_SCALAR],
    domain: tuple[PINT_SCALAR, PINT_SCALAR],
) -> None: ...


@overload
def check_no_times_outside_domain(
    times: Union[NP_ARRAY_OF_FLOAT_OR_INT, NP_FLOAT_OR_INT],
    domain: tuple[NP_FLOAT_OR_INT, NP_FLOAT_OR_INT],
) -> None: ...


def check_no_times_outside_domain(
    times: Union[
        PINT_NUMPY_ARRAY, PINT_SCALAR, NP_ARRAY_OF_FLOAT_OR_INT, NP_FLOAT_OR_INT
    ],
    domain: Union[
        tuple[PINT_SCALAR, PINT_SCALAR], tuple[NP_FLOAT_OR_INT, NP_FLOAT_OR_INT]
    ],
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
    """
    validate_domain(domain)

    times = np.atleast_1d(times)

    outside_domain = np.hstack(
        [
            times[times < domain[0]],
            times[times > domain[1]],
        ]
    )

    if outside_domain.size >= 1:
        msg = (
            f"The {domain=}. "
            "There are time values that are outside this domain: "
            f"{outside_domain=}."
        )
        raise ValueError(msg)
