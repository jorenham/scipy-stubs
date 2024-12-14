from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Self, TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase, sparray
from ._csr import csr_array, csr_matrix
from ._index import IndexMixin
from ._matrix import spmatrix
from ._typing import Index1D, Scalar, ToDTypeBool, ToDTypeComplex, ToDTypeFloat, ToDTypeInt, ToShape2D

__all__ = ["isspmatrix_lil", "lil_array", "lil_matrix"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)

_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

###

class _lil_base(_spbase[_SCT, tuple[int, int]], IndexMixin[_SCT, tuple[int, int]], Generic[_SCT]):
    dtype: np.dtype[_SCT]
    data: onp.Array1D[np.object_]
    rows: onp.Array1D[np.object_]

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
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _lil_base[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _lil_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: ToDTypeBool | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _lil_base[np.int_],
        /,
        arg1: _ToMatrixPy[opt.JustInt],
        shape: ToShape2D | None = None,
        dtype: ToDTypeInt | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _lil_base[np.float64],
        /,
        arg1: _ToMatrixPy[opt.Just[float]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _lil_base[np.complex128],
        /,
        arg1: _ToMatrixPy[opt.Just[complex]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeComplex | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None,
        dtype: _ToDType[_SCT],
        copy: bool = False,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None = None,
        *,
        dtype: _ToDType[_SCT],
        copy: bool = False,
    ) -> None: ...

    #
    @override
    def resize(self, /, *shape: int) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    def getrowview(self, /, i: int) -> Self: ...
    def getrow(self, /, i: onp.ToJustInt) -> csr_array[_SCT, tuple[int, int]] | csr_matrix[_SCT]: ...

class lil_array(_lil_base[_SCT], sparray, Generic[_SCT]):
    @override
    def getrow(self, /, i: onp.ToJustInt) -> csr_array[_SCT, tuple[int, int]]: ...

class lil_matrix(_lil_base[_SCT], spmatrix[_SCT], Generic[_SCT]):  # type: ignore[misc]
    @override
    def getrow(self, /, i: onp.ToJustInt) -> csr_matrix[_SCT]: ...

    # NOTE: using `@override` together with `@overload` causes stubtest to crash...
    @overload  # type: ignore[explicit-override]
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_lil(x: object) -> TypeIs[lil_matrix]: ...
