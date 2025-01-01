# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

from typing import Any, Final
from typing_extensions import deprecated

__all__ = ["isscalarlike", "name", "npfunc", "validateaxis"]

name: Final[str] = ...
npfunc: object = ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def validateaxis(axis: object) -> None: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> Any: ...
