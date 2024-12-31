"""
Support for pretty formatting of our classes

Inspired by:

- the difference between `__repr__` and `__str__` in Python
  (see e.g. https://realpython.com/python-repr-vs-str/)

- the advice from the IPython docs about prettifying output
  (https://ipython.readthedocs.io/en/8.26.0/config/integrating.html#rich-display)

- the way that xarray handles formatting
  (see https://github.com/pydata/xarray/blob/main/xarray/core/formatting.py)

- the way that pint handles formatting
  (see https://github.com/hgrecco/pint/blob/74b708661577623c0c288933d8ed6271f32a4b8b/pint/util.py#L1004)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import IPython.lib.pretty

# Let attrs take care of __repr__


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

    attrs_to_show
        Attributes to include in the string representation.

    Returns
    -------
    :
        Generated string representation of the instance
    """
    instance_type = type(instance).__name__

    attribute_str = [f"{v}={getattr(instance, v)}" for v in attrs_to_show]

    res = f"{instance_type}({', '.join(attribute_str)})"

    return res


def to_pretty(
    instance: Any,
    attrs_to_show: Iterable[str],
    p: IPython.lib.pretty.RepresentationPrinter,
    cycle: bool,
    indent: int = 4,
) -> None:
    """
    Pretty-print an instance using IPython's pretty printer

    Parameters
    ----------
    instance
        Instance to convert

    attrs_to_show
        Attributes to include in the pretty representation.

    p
        Pretty printer

    cycle
        Whether the pretty printer has detected a cycle or not.

    indent
        Indent to apply to the pretty printing group
    """
    instance_type = type(instance).__name__

    with p.group(indent, f"{instance_type}(", ")"):
        for i, att in enumerate(attrs_to_show):
            p.breakable("")
            p.text(f"{att}=")
            p.pretty(getattr(instance, att))

            if i < len(attrs_to_show) - 1:
                p.text(",")
                p.breakable()
