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

        try:
            namespace = config["namespace"]
        except KeyError:
            msg = f"Please specify the namespace to use with {data.name}. {data.path=}"
            raise KeyError(msg)

        member_keys = list(data.members.keys())
        for name in member_keys:
            if name.startswith("_"):
                data.del_member(name)
                continue

            member = data.members[name]
            member.name = f"{namespace}.{name}"

        data.name = namespace

        try:
            return super().render(data, config)
        except Exception:  # pragma: no cover
            print(f"{data.path=}")
            raise
