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
  (see
  [e.g. this line](https://github.com/hgrecco/pint/blob/74b708661577623c0c288933d8ed6271f32a4b8b/pint/util.py#L1004)
  )
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
        String representation of the instance
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


def add_html_attribute_row(
    attribute_name: str, attribute_value_html: str, attribute_rows: list[str]
) -> list[str]:
    """
    Add a row for displaying an attribute's HTML value to a list of existing rows

    Parameters
    ----------
    attribute_name
        Attribute's name

    attribute_value_html
        Attribute's HTML value to display

    attribute_rows
        Existing attribute rows

    Returns
    -------
    :
        Attribute rows, with the new row appended
    """
    attribute_rows.append(
        "<tr>"
        f"<th>{attribute_name}</th>"
        f"<td style='text-align:left;'>{attribute_value_html}</td>"
        "</tr>"
    )

    return attribute_rows


def to_html(
    instance: Any, attrs_to_show: Iterable[str], prefix: str = "continuous_timeseries."
) -> str:
    """
    Convert an instance to its html representation

    Parameters
    ----------
    instance
        Instance to convert

    attrs_to_show
        Attributes to include in the HTML representation.

    prefix
        Prefix to include in front of the instance name when displaying.

    Returns
    -------
    :
        HTML representation of the instance
    """
    instance_type = type(instance).__name__

    header = f"{prefix}{instance_type}"

    attribute_rows: list[str] = []
    for att in attrs_to_show:
        att_val_html = getattr(instance, att)

        try:
            att_val_html = att_val_html._repr_html_()
        except AttributeError:
            att_val_html = str(att_val_html)

        attribute_rows = add_html_attribute_row(att, att_val_html, attribute_rows)

    attribute_rows_for_table = "\n          ".join(attribute_rows)

    css_style = """.continuous-timeseries-wrap {
  /*font-family: monospace;*/
  width: 540px;
}

.continuous-timeseries-header {
  padding: 6px 0 6px 3px;
}

.continuous-timeseries-header > div {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.continuous-timeseries-cls {
  margin-left: 2px;
  margin-right: 10px;
}

.continuous-timeseries-cls {
  font-weight: bold;
}"""

    return "\n".join(
        [
            "<div>",
            "  <style>",
            f"{css_style}",
            "  </style>",
            "  <div class='continuous-timeseries-wrap'>",
            "    <div class='continuous-timeseries-header'>",
            f"      <div class='continuous-timeseries-cls'>{header}</div>",
            "        <table><tbody>",
            f"          {attribute_rows_for_table}",
            "        </tbody></table>",
            "    </div>",
            "  </div>",
            "</div>",
        ]
    )
