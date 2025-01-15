from __future__ import annotations

import griffe


class MyExtension(griffe.Extension):
    def __init__(self, object_paths: list[str] | None = None) -> None:
        self.object_paths = object_paths

    def on_class_instance(
        self,
        node: ast.AST | griffe.ObjectNode,
        cls,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs,
    ) -> None:
        if cls.decorators:
            if (
                cls.decorators[0].callable_path
                == "pandas.api.extensions.register_dataframe_accessor"
            ):
                if len(cls.decorators) > 1:
                    raise NotImplementedError

                cls.name = "pd.DataFrame.ct"

    def on_class_members(
        self,
        node: ast.AST | griffe.ObjectNode,
        cls,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs,
    ) -> None:
        if "ct" in cls.name:
            for key, value in cls.members.items():
                if key == "__init__":
                    continue

                value.name = f"pd.DataFrame.ct.{value.name}"

        if cls.decorators:
            if (
                cls.decorators[0].callable_path
                == "pandas.api.extensions.register_dataframe_accessor"
            ):
                if len(cls.decorators) > 1:
                    raise NotImplementedError

                cls.name = "pd.DataFrame.ct"
