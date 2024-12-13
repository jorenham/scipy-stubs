from types import EllipsisType
from typing import Any, Generic, TypeAlias, overload
from typing_extensions import Buffer, Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import _spbase
from ._typing import Complex, Float, Int, Scalar

_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=Scalar, covariant=True)

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
    | _spbase[np.bool_]
)

###

INT_TYPES: tuple[type[int], type[np.integer[Any]]] = ...

class IndexMixin(Generic[_SCT_co]):
    @overload
    def __getitem__(self, ix: _ToIndex2D, /) -> _SCT_co: ...
    @overload
    def __getitem__(self, ixs: _ToSlice2D, /) -> Self: ...

    #
    @overload
    def __setitem__(self: IndexMixin[Int], key: _ToIndex2D, x: _ToInt, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Float], key: _ToIndex2D, x: _ToFloat, /) -> None: ...
    @overload
    def __setitem__(self: IndexMixin[Complex], key: _ToIndex2D, x: _ToComplex, /) -> None: ...
    @overload
    def __setitem__(self, key: _ToIndex2D | _ToSlice2D, x: object, /) -> None: ...
