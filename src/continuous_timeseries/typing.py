"""
Helpful type hints
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pint.facets.numpy.quantity
from typing_extensions import TypeAlias

PINT_SCALAR: TypeAlias = pint.facets.numpy.quantity.NumpyQuantity[np.number[Any]]
"""
Type alias for a pint quantity that wraps a numpy scalar
"""

PINT_NUMPY_ARRAY: TypeAlias = pint.facets.numpy.quantity.NumpyQuantity[
    npt.NDArray[np.number[Any]]
]
"""
Type alias for a pint quantity that wraps a numpy array

No shape hints because that doesn't seem to be supported by numpy yet.
"""
