# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

from warnings import warn
from typing import Any
from typing_extensions import deprecated

from . import _bsr, _matrix

__all__ = [
    "bsr_matmat",
    "bsr_matrix",
    "bsr_matvec",
    "bsr_matvecs",
    "bsr_sort_indices",
    "bsr_tocsr",
    "bsr_transpose",
    "check_shape",
    "csr_matmat_maxnnz",
    "getdata",
    "getdtype",
    "isshape",
    "isspmatrix_bsr",
    "spmatrix",
    "to_native",
    "upcast",
    "warn",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class bsr_matrix(_bsr.bsr_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def bsr_matmat(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_matvec(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_matvecs(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_sort_indices(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_tocsr(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_transpose(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_matmat_maxnnz(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_bsr(x: object) -> Any: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def to_native(A: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdata(obj: object, dtype: object = ..., copy: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_nd: object = ...) -> Any: ...
