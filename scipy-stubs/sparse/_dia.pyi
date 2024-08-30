from typing_extensions import override

from scipy._typing import Untyped
from ._base import sparray, spmatrix
from ._data import _data_matrix

__all__ = ["dia_array", "dia_matrix", "isspmatrix_dia"]

class _dia_base(_data_matrix):
    data: Untyped
    offsets: Untyped
    def __init__(
        self,
        arg1: Untyped,
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
        *,
        maxprint: Untyped | None = None,
    ) -> None: ...
    @override
    def resize(self, *shape: int): ...  # type: ignore[override]

class dia_array(_dia_base, sparray): ...
class dia_matrix(spmatrix, _dia_base): ...

def isspmatrix_dia(x: Untyped) -> bool: ...
