# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

from typing import Any, Final
from typing_extensions import deprecated

from . import _base, _matrix

__all__ = [
    "MAXPRINT",
    "SparseEfficiencyWarning",
    "SparseFormatWarning",
    "SparseWarning",
    "asmatrix",
    "check_reshape_kwargs",
    "check_shape",
    "get_sum_dtype",
    "isdense",
    "isscalarlike",
    "issparse",
    "isspmatrix",
    "spmatrix",
    "validateaxis",
]

MAXPRINT: Final[int] = ...

@deprecated("will be removed in SciPy v2.0.0")
class SparseWarning(_base.SparseWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class SparseFormatWarning(_base.SparseFormatWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class SparseEfficiencyWarning(_base.SparseEfficiencyWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def issparse(x: object) -> bool: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix(x: object) -> bool: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_reshape_kwargs(kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def validateaxis(axis: object) -> None: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def asmatrix(data: object, dtype: object = ...) -> Any: ...
