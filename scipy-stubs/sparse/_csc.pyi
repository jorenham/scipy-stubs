from typing import Any, Generic, Literal
from typing_extensions import TypeIs, TypeVar, override

from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Scalar

__all__ = ["csc_array", "csc_matrix", "isspmatrix_csc"]

_SCT = TypeVar("_SCT", bound=Scalar, default=Any)

###

class _csc_base(_cs_matrix[_SCT], Generic[_SCT]):
    @property
    @override
    def format(self, /) -> Literal["csc"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

class csc_array(_csc_base[_SCT], sparray, Generic[_SCT]): ...
class csc_matrix(spmatrix[_SCT], _csc_base[_SCT], Generic[_SCT]): ...

def isspmatrix_csc(x: object) -> TypeIs[csc_matrix]: ...
