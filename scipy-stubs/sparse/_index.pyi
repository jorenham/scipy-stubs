from types import EllipsisType
from typing import Any, Generic, TypeAlias, overload
from typing_extensions import Buffer, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import _spbase
from ._typing import Complex, Float, Int, Scalar

_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int, int], covariant=True)

_Self1DT = TypeVar("_Self1DT", bound=IndexMixin[Any, tuple[int]])
_Self2DT = TypeVar("_Self2DT", bound=IndexMixin[Any, tuple[int, int]])

_ToInt: TypeAlias = str | op.CanInt | op.CanIndex | op.CanTrunc | Buffer
_ToFloat: TypeAlias = str | op.CanFloat | op.CanIndex | Buffer
_ToComplex: TypeAlias = str | op.CanComplex | op.CanFloat | op.CanIndex | complex

_ToIndex1D: TypeAlias = op.CanIndex | tuple[op.CanIndex]
_ToIndex2D: TypeAlias = tuple[op.CanIndex, op.CanIndex]

_ToSlice1D: TypeAlias = slice | EllipsisType | onp.ToJustInt2D | onp.ToBool1D
_ToSlice2D: TypeAlias = (
    slice
    | EllipsisType
    | tuple[_ToIndex1D | _ToSlice1D, _ToSlice1D]
    | tuple[_ToSlice1D, _ToSlice1D | _ToIndex1D]
    | onp.ToBool2D
    | list[bool]
    | list[np.bool_]
    | list[int]
    | _spbase[np.bool_, tuple[int, int]]
)

###

INT_TYPES: tuple[type[int], type[np.integer[Any]]] = ...

class IndexMixin(Generic[_SCT, _ShapeT_co]):
    @overload
    def __getitem__(self: IndexMixin[Any, tuple[int]], ix: op.CanIndex, /) -> _SCT: ...
    @overload
    def __getitem__(self: IndexMixin[Any, tuple[int, int]], ix: _ToIndex2D, /) -> _SCT: ...
    @overload
    def __getitem__(self: _Self1DT, ixs: _ToSlice1D, /) -> _Self1DT: ...
    @overload
    def __getitem__(self: _Self2DT, ixs: _ToSlice2D, /) -> _Self2DT: ...

    #
    @overload
    def __setitem__(self: IndexMixin[Any, tuple[int]], key: op.CanIndex | _ToSlice1D, x: _SCT, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Any], key: _ToIndex2D | _ToSlice2D, x: _SCT, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Int, tuple[int]], key: _ToIndex1D, x: _ToInt, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Int], key: _ToIndex2D, x: _ToInt, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Float, tuple[int]], key: _ToIndex1D, x: _ToFloat, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Float], key: _ToIndex2D, x: _ToFloat, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Complex, tuple[int]], key: _ToIndex1D, x: _ToComplex, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Complex], key: _ToIndex2D, x: _ToComplex, /) -> None: ...
    @overload
    def __setitem__(self, key: _ToIndex1D | _ToSlice1D | _ToIndex2D | _ToSlice2D, x: object, /) -> None: ...
