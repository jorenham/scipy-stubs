from typing import Any, Generic, Literal, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric

__all__ = ["csc_array", "csc_matrix", "isspmatrix_csc"]

_SCT = TypeVar("_SCT", bound=Numeric, default=Any)

###

class _csc_base(_cs_matrix[_SCT, tuple[int, int]], Generic[_SCT]):
    @property
    @override
    def format(self, /) -> Literal["csc"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    @overload  # type: ignore[explicit-override]
    def count_nonzero(self, /, axis: None = None) -> int: ...
    @overload
    def count_nonzero(self, /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...

class csc_array(_csc_base[_SCT], sparray[_SCT, tuple[int, int]], Generic[_SCT]): ...

class csc_matrix(_csc_base[_SCT], spmatrix[_SCT], Generic[_SCT]):  # type: ignore[misc]
    # NOTE: using `@override` together with `@overload` causes stubtest to crash...
    @overload  # type: ignore[explicit-override]
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csc(x: object) -> TypeIs[csc_matrix]: ...
