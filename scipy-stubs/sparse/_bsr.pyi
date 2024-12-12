from typing import Literal
from typing_extensions import override

import optype as op
from scipy._typing import Untyped
from ._base import sparray
from ._compressed import _cs_matrix
from ._data import _minmax_mixin
from ._matrix import spmatrix

__all__ = ["bsr_array", "bsr_matrix", "isspmatrix_bsr"]

# TODO(jorenham): generic dtype
class _bsr_base(_cs_matrix, _minmax_mixin):
    data: Untyped
    indices: Untyped
    indptr: Untyped

    @property
    @override
    def format(self, /) -> Literal["bsr"]: ...

    #
    @property
    def blocksize(self, /) -> tuple[int, int]: ...

    #
    def __init__(
        self,
        /,
        arg1: Untyped,
        shape: tuple[op.CanIndex, op.CanIndex] | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...

class bsr_array(_bsr_base, sparray): ...
class bsr_matrix(spmatrix, _bsr_base): ...

def isspmatrix_bsr(x: Untyped) -> bool: ...
