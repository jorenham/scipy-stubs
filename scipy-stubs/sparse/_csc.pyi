from typing_extensions import override

from scipy._typing import Untyped
from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix

__all__ = ["csc_array", "csc_matrix", "isspmatrix_csc"]

class _csc_base(_cs_matrix): ...
class csc_array(_csc_base, sparray): ...
class csc_matrix(spmatrix, _csc_base): ...

def isspmatrix_csc(x: Untyped) -> bool: ...
