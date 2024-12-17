# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

from . import _base, _csc

__all__ = ["csc_matrix", "csc_tocsr", "expandptr", "isspmatrix_csc", "spmatrix", "upcast"]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix(_base.spmatrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class csc_matrix(_csc.csc_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_csc(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def csc_tocsr(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def expandptr(*args: object, **kwargs: object) -> object: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> object: ...
