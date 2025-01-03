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

import textwrap
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, cast

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
    attrs_to_show: Collection[str],
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
            p.breakable("")  # type: ignore
            p.text(f"{att}=")  # type: ignore
            p.pretty(getattr(instance, att))  # type: ignore

            if i < len(attrs_to_show) - 1:
                p.text(",")  # type: ignore


def get_html_repr_safe(
    instance: Any,
    escapes: tuple[tuple[str, str], ...] = (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
    ),
) -> str:
    """
    Get the HTML representation of an instance

    Parameters
    ----------
    instance
        Instance of which to get the HTML representation

    escapes
        Escape characters to apply when a raw string has to be used

    Returns
    -------
    :
        HTML representation.

        If `instance` has a `_repr_html_` method, this is used.
        Otherwise the string representation of `instance` is returned.
    """
    # Workaround to make our domain render nicely
    if isinstance(instance, (tuple, list)):
        elements_html = [get_html_repr_safe(v, escapes=escapes) for v in instance]
        elements_comma_separated = ", ".join(elements_html)

        if isinstance(instance, tuple):
            brackets = "()"

        if isinstance(instance, list):
            brackets = "[]"

        return f"{brackets[0]}{elements_comma_separated}{brackets[1]}"

    try:
        repr_html = cast(str, instance._repr_html_())
    except AttributeError:
        repr_html = str(instance)

        for raw, escaped in escapes:
            repr_html = repr_html.replace(raw, escaped)

    return repr_html


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
        "<tr>\n"
        f"  <th>{attribute_name}</th>\n"
        "  <td style='text-align:left;'>\n"
        f"{textwrap.indent(attribute_value_html, '    ')}\n"
        "  </td>\n"
        "</tr>"
    )

    return attribute_rows


def make_html_attribute_table(attribute_rows: Iterable[str]) -> str:
    """
    Make an HTML table of attributes

    Parameters
    ----------
    attribute_rows
        Attribute rows to put in the table

    Returns
    -------
    :
        HTML table of attribute rows
    """
    attribute_rows_for_table = textwrap.indent("\n".join(attribute_rows), "  ")
    html_l = [
        "<table><tbody>",
        f"{attribute_rows_for_table}",
        "</tbody></table>",
    ]

    return "\n".join(html_l)


def apply_ct_html_styling(display_name: str, attribute_table: str) -> str:
    """
    Apply continuous timeseries' HTML styling for displaying an instance

    Parameters
    ----------
    display_name
        Name to display as the instance's name

        This appears as the title of the output table/div

    attribute_table
        Table of attributes of the instance.

        Generally created with [`make_html_attribute_table`][(m)].

    Returns
    -------
    :
        Formatted HTML with CSS styling included
    """
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
    html_l = [
        "<div>",
        "  <style>",
        f"{css_style}",
        "  </style>",
        "  <div class='continuous-timeseries-wrap'>",
        "    <div class='continuous-timeseries-header'>",
        f"      <div class='continuous-timeseries-cls'>{display_name}</div>",
        textwrap.indent(attribute_table, "      "),
        "    </div>",
        "  </div>",
        "</div>",
    ]

    return "\n".join(html_l)


def to_html(
    instance: Any,
    attrs_to_show: Iterable[str],
    prefix: str = "continuous_timeseries.",
    include_header: bool = True,
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

    include_header
        Should the header be included when formatting the object?

    Returns
    -------
    :
        HTML representation of the instance
    """
    instance_type = type(instance).__name__

    header = f"{prefix}{instance_type}"

    attribute_rows: list[str] = []
    for att in attrs_to_show:
        att_val = getattr(instance, att)

        if hasattr(att_val, "_repr_html_internal_row_"):
            # One of our objects
            att_val_html = att_val._repr_html_internal_row_()

        else:
            att_val_html = get_html_repr_safe(att_val)

        attribute_rows = add_html_attribute_row(att, att_val_html, attribute_rows)

    attribute_table = make_html_attribute_table(attribute_rows)
    if not include_header:
        return attribute_table

    return apply_ct_html_styling(display_name=header, attribute_table=attribute_table)
