from typing import Any, Generic, Literal, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

###

class _csr_base(_cs_matrix[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def format(self, /) -> Literal["csr"]: ...

    #
    @overload  # type: ignore[explicit-override]
    def count_nonzero(self, /, axis: None = None) -> int: ...
    @overload
    def count_nonzero(self: _csr_base[Any, tuple[int]], /, axis: op.CanIndex) -> int: ...
    @overload
    def count_nonzero(self: _csr_base[Any, tuple[int, int]], /, axis: op.CanIndex) -> onp.Array1D[np.int32 | np.int64]: ...
    @overload
    def count_nonzero(self: csr_array, /, axis: op.CanIndex) -> int | onp.Array1D[np.int32 | np.int64]: ...  # type: ignore[misc]

class csr_array(_csr_base[_SCT, _ShapeT_co], sparray[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]): ...

class csr_matrix(_csr_base[_SCT, tuple[int, int]], spmatrix[_SCT], Generic[_SCT]):  # type: ignore[misc]
    # NOTE: using `@override` together with `@overload` causes stubtest to crash...
    @overload  # type: ignore[explicit-override]
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
