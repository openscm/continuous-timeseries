"""
Implementation of python_accessors handler
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import griffe
from mkdocstrings.loggers import get_logger
from mkdocstrings_handlers.python_xref.handler import PythonRelXRefHandler

__all__ = ["PythonAccessorHandler"]

logger = get_logger(__name__)


class PythonAccessorHandler(PythonRelXRefHandler):
    """
    Extended version of mkdocstrings Python handler

    * Converts references so that they appear in their accessor namespace
    """

    def render(self, data: griffe.Object, config: Mapping[str, Any]) -> str:
        if not isinstance(data, griffe.Class):
            raise NotImplementedError(data)

        if len(data.decorators) != 1:
            raise NotImplementedError(data.decorators)

        decorator_act = data.decorators[0].callable_path
        decorator_exp = "pandas.api.extensions.register_dataframe_accessor"
        class_being_accessed = "pd.DataFrame"
        if decorator_act != decorator_exp:
            raise NotImplementedError(decorator_act)

        accessor_key = (
            data.decorators[0].value.arguments[0].replace("'", "").replace('"', "")
        )
        for name, member in data.members.items():
            if name.startswith("_"):
                continue

            member.name = f"{class_being_accessed}.{accessor_key}.{name}"

        data.name = f"{class_being_accessed}.{accessor_key}"

        try:
            return super().render(data, config)
        except Exception:  # pragma: no cover
            print(f"{data.path=}")
            raise
