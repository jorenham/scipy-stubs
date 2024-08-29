from typing_extensions import override

from scipy._typing import Untyped

from ._base import issparse as issparse, sparray
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
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: Untyped | None = None,
    ) -> None: ...
    @property
    def blocksize(self) -> tuple[int, int]: ...
    @override
    def diagonal(self, k: int = 0) -> Untyped: ...
    @override
    def __getitem__(self, key: Untyped) -> Untyped: ...
    @override
    def __setitem__(self, key: Untyped, val: Untyped) -> None: ...
    @override
    def tobsr(self, blocksize: Untyped | None = None, copy: bool = False) -> Untyped: ...
    @override
    def tocsr(self, copy: bool = False) -> Untyped: ...
    @override
    def tocsc(self, copy: bool = False) -> Untyped: ...
    @override
    def tocoo(self, copy: bool = True) -> Untyped: ...
    @override
    def toarray(self, order: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    @override
    def transpose(self, axes: Untyped | None = None, copy: bool = False) -> Untyped: ...

class bsr_array(_bsr_base, sparray): ...
class bsr_matrix(spmatrix, _bsr_base): ...

def isspmatrix_bsr(x: Untyped) -> bool: ...
