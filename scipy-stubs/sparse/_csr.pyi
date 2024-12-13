from typing import Any, Generic, Literal
from typing_extensions import TypeIs, TypeVar, override

from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Scalar

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

_SCT = TypeVar("_SCT", bound=Scalar, default=Any)

###

class _csr_base(_cs_matrix[_SCT], Generic[_SCT]):
    @property
    @override
    def format(self, /) -> Literal["csr"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

class csr_array(_csr_base[_SCT], sparray, Generic[_SCT]): ...
class csr_matrix(spmatrix[_SCT], _csr_base[_SCT], Generic[_SCT]): ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
