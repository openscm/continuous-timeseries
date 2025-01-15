"""Python handler for mkdocstrings."""

from .handler import PythonAccessorHandler

__all__ = ["get_handler"]

get_handler = PythonAccessorHandler
