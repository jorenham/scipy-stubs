# This module is not meant for public use and will be removed in SciPy v2.0.0.
import operator
import sys
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

if sys.version_info >= (3, 12):
    @deprecated("will be removed in SciPy v2.0.0")
    def warn(
        message: object,
        category: object = ...,
        stacklevel: object = ...,
        source: object = ...,
        *,
        skip_file_prefixes: object = ...,
    ) -> None: ...
else:
    @deprecated("will be removed in SciPy v2.0.0")
    def warn(message: object, category: object = ..., stacklevel: object = ..., source: object = ...) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
def csr_column_index1(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_column_index2(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_row_index(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_row_slice(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_sample_offsets(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_sample_values(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_todense(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_csr_submatrix(*args: object, **kwargs: object) -> object: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def upcast_char(*args: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def to_native(A: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def downcast_intp_index(arr: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isintlike(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def is_pydata_spmatrix(m: object) -> object: ...
