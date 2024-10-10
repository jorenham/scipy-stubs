# This file is not meant for public use and will be removed in SciPy v2.0.0.

from collections.abc import Callable
from typing_extensions import TypeVar, deprecated

import optype as op

_F = TypeVar("_F", bound=Callable[..., object])

__all__ = [
    "docformat",
    "extend_notes_in_docstring",
    "filldoc",
    "indentcount_lines",
    "inherit_docstring_from",
    "replace_notes_in_docstring",
    "unindent_dict",
    "unindent_string",
]

@deprecated("will be removed in SciPy v2.0.0")
def docformat(docstring: str, docdict: dict[str, str] | None = None) -> str: ...
@deprecated("will be removed in SciPy v2.0.0")
def inherit_docstring_from(cls: type | object) -> Callable[[_F], _F]: ...
@deprecated("will be removed in SciPy v2.0.0")
def extend_notes_in_docstring(cls: type | object, notes: str) -> Callable[[_F], _F]: ...
@deprecated("will be removed in SciPy v2.0.0")
def replace_notes_in_docstring(cls: type | object, notes: str) -> Callable[[_F], _F]: ...
@deprecated("will be removed in SciPy v2.0.0")
def indentcount_lines(lines: op.CanIter[op.CanNext[str]]) -> int: ...
@deprecated("will be removed in SciPy v2.0.0")
def filldoc(docdict: dict[str, str], unindent_params: bool = True) -> Callable[[_F], _F]: ...
@deprecated("will be removed in SciPy v2.0.0")
def unindent_dict(docdict: dict[str, str]) -> dict[str, str]: ...
@deprecated("will be removed in SciPy v2.0.0")
def unindent_string(docstring: str) -> str: ...
