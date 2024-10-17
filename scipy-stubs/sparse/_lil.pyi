from typing_extensions import override

from scipy._typing import Untyped
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix

__all__ = ["isspmatrix_lil", "lil_array", "lil_matrix"]

class _lil_base(_spbase, IndexMixin):
    dtype: Untyped
    rows: Untyped
    data: Untyped
    def __init__(
        self,
        arg1: Untyped,
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
    ) -> None: ...
    def getrowview(self, i: int) -> Untyped: ...
    def getrow(self, i: int) -> Untyped: ...
    @override
    def resize(self, *shape: int) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def count_nonzero(self) -> int: ...

def isspmatrix_lil(x: Untyped) -> Untyped: ...

class lil_array(_lil_base, sparray): ...
class lil_matrix(spmatrix, _lil_base): ...
