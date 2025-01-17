"""
Griffe extension for handling extension documentation
"""

from __future__ import annotations

import ast
from typing import Any

import griffe


class AccessorNamespace(griffe.Extension):
    """
    Extension for applying an accessor namespace to an accessor class
    """

    def __init__(self, class_to_apply_to: str, namespace: str) -> None:
        self.class_to_apply_to = class_to_apply_to
        self.namespace = namespace

    def on_class_members(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        cls: griffe.Class,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        """
        Update after class members have finished being updated
        """
        if cls.name != self.class_to_apply_to:
            return  # Only apply to selected class

        # Set overall name
        cls.name = self.namespace

        # Then update names for individual members
        member_keys = list(cls.members.keys())
        for name in member_keys:
            if name.startswith("_"):
                # Don't document hidden methods etc.
                # Could make this configuration instead
                # if real customisation was needed.
                cls.del_member(name)
                continue

            cls.members[name].name = f"{self.namespace}.{name}"
