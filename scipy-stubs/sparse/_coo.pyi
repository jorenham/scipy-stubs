# mypy: disable-error-code="explicit-override"

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._matrix import spmatrix
from ._typing import Index1D, Int, Scalar, ToShape, ToShape1D, ToShape2D

__all__ = ["coo_array", "coo_matrix", "isspmatrix_coo"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]
_ToData: TypeAlias = tuple[onp.ArrayND[_SCT], tuple[onp.ArrayND[Int]] | tuple[onp.ArrayND[Int], onp.ArrayND[Int]]]

###

class _coo_base(_data_matrix[_SCT, _ShapeT_co], _minmax_mixin[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    data: onp.Array1D[_SCT]
    coords: tuple[Index1D] | tuple[Index1D, Index1D]
    has_canonical_format: bool

    @property
    @override
    def format(self, /) -> Literal["coo"]: ...
    #
    @property
    @override
    def shape(self, /) -> _ShapeT_co: ...
    #
    @property
    def row(self, /) -> Index1D: ...
    @row.setter
    def row(self, new_row: onp.ToInt1D, /) -> None: ...
    #
    @property
    def col(self, /) -> Index1D: ...
    @col.setter
    def col(self, new_col: onp.ToInt1D, /) -> None: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape | None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: None
    def __init__(  # type: ignore[misc]
        self: coo_array[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _coo_base[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _coo_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _coo_base[np.int_],
        /,
        arg1: _ToMatrixPy[opt.JustInt],
        shape: ToShape | None = None,
        dtype: type[opt.JustInt] | onp.AnyIntPDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _coo_base[np.float64],
        /,
        arg1: _ToMatrixPy[opt.Just[float]],
        shape: ToShape | None = None,
        dtype: type[opt.Just[float]] | onp.AnyFloat64DType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _coo_base[np.complex128],
        /,
        arg1: _ToMatrixPy[opt.Just[complex]],
        shape: ToShape | None = None,
        dtype: type[opt.Just[complex]] | onp.AnyComplex128DType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape | None,
        dtype: _ToDType[_SCT],
        copy: bool = False,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape | None = None,
        *,
        dtype: _ToDType[_SCT],
        copy: bool = False,
    ) -> None: ...

    #
    def sum_duplicates(self, /) -> None: ...
    def eliminate_zeros(self, /) -> None: ...

class coo_array(_coo_base[_SCT, _ShapeT_co], sparray, Generic[_SCT, _ShapeT_co]): ...

class coo_matrix(_coo_base[_SCT, tuple[int, int]], spmatrix[_SCT], Generic[_SCT]):  # type: ignore[misc]
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    #
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_coo(x: object) -> TypeIs[coo_matrix]: ...
