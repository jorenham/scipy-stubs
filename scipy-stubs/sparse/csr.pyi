# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

from typing import Any
from typing_extensions import deprecated

from . import _csr, _matrix

__all__ = [
    "csr_count_blocks",
    "csr_matrix",
    "csr_tobsr",
    "csr_tocsc",
    "get_csr_submatrix",
    "isspmatrix_csr",
    "spmatrix",
    "upcast",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_matrix.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class csr_matrix(_csr.csr_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_csr(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_count_blocks(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_tobsr(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def csr_tocsc(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_csr_submatrix(*args: object, **kwargs: object) -> Any: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...
