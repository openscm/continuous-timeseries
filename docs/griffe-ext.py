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

    def __init__(
        self,
        class_to_apply_to: str,
        namespace: str,
    ) -> None:
        self.class_to_apply_to = class_to_apply_to
        self.namespace = namespace

    # def on_node(
    #     self,
    #     *,
    #     node: ast.AST | griffe.ObjectNode,
    #     agent: griffe.Visitor | griffe.Inspector,
    #     **kwargs: Any,
    # ) -> None:
    #     breakpoint()

    def on_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        obj: griffe.Object,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        modules = self.class_to_apply_to.split(".")[:-1]
        class_of_interest = self.class_to_apply_to.split(".")[0]

        if isinstance(obj, griffe.Module) and obj.name in modules:
            breakpoint()

        if isinstance(obj, griffe.Class) and obj.name == class_of_interest:
            breakpoint()

    def on_package_loaded(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        """
        Update after a full package has been loaded
        """
        # if not pkg.name.startswith(self.class_to_apply_to_root_package):
        #     # Not the package we're interested in, do nothing
        #     return
        #
        # # Shift the loaded package to the namespace we want it to be in.
        # breakpoint()
        # loader.modules_collection.set_member(
        #     self.namespace,
        #     loader.modules_collection.get_member(self.class_to_apply_to_root_package),
        # )

        # Delete what it thinks it has loaded from modules_collection
        # so it knows to re-load this module next time.
        loader.modules_collection.del_member(self.class_to_apply_to_root_package)

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
            # Not the class we're interested in, do nothing
            return

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
