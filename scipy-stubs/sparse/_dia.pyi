from typing import Literal
from typing_extensions import override

import optype as op
from scipy._typing import Untyped
from ._base import sparray
from ._data import _data_matrix
from ._matrix import spmatrix

__all__ = ["dia_array", "dia_matrix", "isspmatrix_dia"]

# TODO(jorenham): generic dtype
class _dia_base(_data_matrix):
    data: Untyped
    offsets: Untyped

    @property
    @override
    def format(self, /) -> Literal["dia"]: ...
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

class dia_array(_dia_base, sparray): ...
class dia_matrix(spmatrix, _dia_base): ...

def isspmatrix_dia(x: Untyped) -> bool: ...
