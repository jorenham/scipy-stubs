# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing import Final
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

MAXPRINT: Final[int]

@deprecated("will be removed in SciPy v2.0.0")
class SparseWarning(_base.SparseWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class SparseFormatWarning(_base.SparseFormatWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class SparseEfficiencyWarning(_base.SparseEfficiencyWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def issparse(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix(x: object) -> object: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_reshape_kwargs(kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def validateaxis(axis: object) -> None: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def asmatrix(data: object, dtype: object = ...) -> object: ...
