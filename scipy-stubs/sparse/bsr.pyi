# This module is not meant for public use and will be removed in SciPy v2.0.0.
import sys
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
def bsr_matmat(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_matvec(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_matvecs(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_sort_indices(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_tocsr(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def bsr_transpose(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_matmat_maxnnz(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_bsr(x: object) -> object: ...

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

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def to_native(A: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdata(obj: object, dtype: object = ..., copy: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
