from typing import Literal
from typing_extensions import override

import optype as op
from scipy._typing import Untyped
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix

__all__ = ["isspmatrix_lil", "lil_array", "lil_matrix"]

# TODO(jorenham): generic dtype
class _lil_base(_spbase, IndexMixin):
    dtype: Untyped
    rows: Untyped
    data: Untyped

    @property
    @override
    def format(self, /) -> Literal["lil"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    def __init__(
        self,
        /,
        arg1: Untyped,
        shape: tuple[op.CanIndex, op.CanIndex] | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
    ) -> None: ...

    #
    @override
    def resize(self, /, *shape: int) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    def getrowview(self, /, i: int) -> Untyped: ...
    def getrow(self, /, i: int) -> Untyped: ...

class lil_array(_lil_base, sparray): ...
class lil_matrix(spmatrix, _lil_base): ...

def isspmatrix_lil(x: Untyped) -> Untyped: ...
