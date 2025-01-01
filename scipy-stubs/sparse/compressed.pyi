# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

import operator
from warnings import warn
from typing import Any
from typing_extensions import deprecated

from . import _base, _index

__all__ = [
    "IndexMixin",
    "SparseEfficiencyWarning",
    "check_shape",
    "csr_column_index1",
    "csr_column_index2",
    "csr_row_index",
    "csr_row_slice",
    "csr_sample_offsets",
    "csr_sample_values",
    "csr_todense",
    "downcast_intp_index",
    "get_csr_submatrix",
    "get_sum_dtype",
    "getdtype",
    "is_pydata_spmatrix",
    "isdense",
    "isintlike",
    "isscalarlike",
    "isshape",
    "operator",
    "to_native",
    "upcast",
    "upcast_char",
    "warn",
]

@deprecated("will be removed in SciPy v2.0.0")
class SparseEfficiencyWarning(_base.SparseEfficiencyWarning): ...

@deprecated("will be removed in SciPy v2.0.0")
class IndexMixin(_index.IndexMixin): ...

@deprecated("will be removed in SciPy v2.0.0")
def csr_column_index1(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_column_index2(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_row_index(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_row_slice(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_sample_offsets(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_sample_values(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_todense(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_csr_submatrix(*args: object, **kwargs: object) -> Any: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def upcast_char(*args: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def to_native(A: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_nd: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def downcast_intp_index(arr: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isintlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def is_pydata_spmatrix(m: object) -> Any: ...
