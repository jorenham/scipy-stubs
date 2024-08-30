from collections.abc import Sequence

import numpy.typing as npt
import scipy._typing as spt
from scipy._typing import Untyped
from ._base import sparray
from ._compressed import _cs_matrix
from ._data import _minmax_mixin
from ._matrix import spmatrix

__all__ = ["bsr_array", "bsr_matrix", "isspmatrix_bsr"]

class _bsr_base(_cs_matrix, _minmax_mixin):
    data: Untyped
    indices: Untyped
    indptr: Untyped
    def __init__(
        self,
        arg1: Untyped,
        shape: spt.AnyInt | Sequence[spt.AnyInt] | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: Untyped | None = None,
    ) -> None: ...
    @property
    def blocksize(self) -> tuple[int, int]: ...

class bsr_array(_bsr_base, sparray): ...
class bsr_matrix(spmatrix, _bsr_base): ...

def isspmatrix_bsr(x: Untyped) -> bool: ...
