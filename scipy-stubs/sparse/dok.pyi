# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

import itertools
from typing import Any
from typing_extensions import deprecated

from . import _dok, _index, _matrix

__all__ = [
    "IndexMixin",
    "check_shape",
    "dok_matrix",
    "getdtype",
    "isdense",
    "isintlike",
    "isscalarlike",
    "isshape",
    "isspmatrix_dok",
    "itertools",
    "spmatrix",
    "upcast",
    "upcast_scalar",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class dok_matrix(_dok.dok_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class IndexMixin(_index.IndexMixin): ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def upcast_scalar(dtype: object, scalar: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isintlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_dok(x: object) -> Any: ...
