# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

from . import _dia, _matrix

__all__ = [
    "check_shape",
    "dia_matrix",
    "dia_matvec",
    "get_sum_dtype",
    "getdtype",
    "isshape",
    "isspmatrix_dia",
    "spmatrix",
    "upcast_char",
    "validateaxis",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class dia_matrix(_dia.dia_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def dia_matvec(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_dia(x: object) -> object: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast_char(*args: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def validateaxis(axis: object) -> None: ...
