from typing_extensions import override

from scipy._typing import Untyped
from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

class _csr_base(_cs_matrix): ...
class csr_array(_csr_base, sparray): ...
class csr_matrix(spmatrix, _csr_base): ...

def isspmatrix_csr(x: Untyped) -> bool: ...
