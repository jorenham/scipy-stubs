# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

import operator
from warnings import warn
from typing import Any, type_check_only
from typing_extensions import deprecated

from . import _base, _coo, _matrix

__all__ = [
    "SparseEfficiencyWarning",
    "check_reshape_kwargs",
    "check_shape",
    "coo_matrix",
    "coo_matvec",
    "coo_tocsr",
    "coo_todense",
    "downcast_intp_index",
    "getdata",
    "getdtype",
    "isshape",
    "isspmatrix_coo",
    "operator",
    "spmatrix",
    "to_native",
    "upcast",
    "upcast_char",
    "warn",
]

@deprecated("will be removed in SciPy v2.0.0")
class SparseEfficiencyWarning(_base.SparseEfficiencyWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class coo_matrix(_coo.coo_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_coo(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def coo_matvec(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def coo_tocsr(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def coo_todense(*args: object, **kwargs: object) -> Any: ...

# sputils
@type_check_only
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...  # does not exist, but is exported anyway...?
@deprecated("will be removed in SciPy v2.0.0")
def upcast_char(*args: object) -> Any: ...
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
@deprecated("will be removed in SciPy v2.0.0")
def check_reshape_kwargs(kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def downcast_intp_index(arr: object) -> Any: ...
