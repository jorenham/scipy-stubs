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
        *,
        maxprint: Untyped | None = None,
    ) -> None: ...
    def getrowview(self, i) -> Untyped: ...
    def getrow(self, i) -> Untyped: ...
    @override
    def resize(self, *shape: int): ...  # type: ignore[override]

def isspmatrix_lil(x) -> Untyped: ...

class lil_array(_lil_base, sparray): ...
class lil_matrix(spmatrix, _lil_base): ...
