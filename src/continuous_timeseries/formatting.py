"""
Support for pretty formatting of our classes

Inspired by [xarray](https://github.com/pydata/xarray/blob/main/xarray/core/formatting.py)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def to_str(instance: Any, attrs_to_show: Iterable[str]) -> str:
    """
    Convert an instance to its string representation

    Only include specified attributes in the representation.
    Show the string representation of the attributes to show.

    As a note, the point of this is to provide a helpful representation for users.
    The `repr` representation is intended for developers.

    For more details, see e.g. https://realpython.com/python-repr-vs-str/

    Parameters
    ----------
    instance
        Instance to convert to str

    exposed_attributes
        Attributes to include in the string representation.

    Returns
    -------
    :
        Generated string representation of the instance
    """
    instance_type = type(instance).__name__

    attribute_str = [f"{v.name}={getattr(instance, v.name)}" for v in attrs_to_show]

    res = f"{instance_type}({', '.join(attribute_str)})"

    return res
