import abc
from collections.abc import Sequence
from typing import Any, Generic, Literal, overload
from typing_extensions import Never, Self, TypeVar, override

import numpy as np
import optype.numpy as onp
from ._base import _spbase, sparray
from ._coo import coo_array, coo_matrix
from ._matrix import spmatrix
from ._sputils import _ScalarLike
from ._typing import Numeric, ToShape1D, ToShape2D

__all__: list[str] = []

_SCT = TypeVar("_SCT", bound=Numeric)
_SCT_co = TypeVar("_SCT_co", bound=Numeric, default=Numeric, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=onp.AtLeast1D, default=onp.AtLeast1D, covariant=True)

###

class _data_matrix(_spbase[_SCT_co, _ShapeT_co], Generic[_SCT_co, _ShapeT_co]):
    @property
    @abc.abstractmethod
    @override
    def format(self, /) -> Literal["bsr", "coo", "csc", "csr", "dia"]: ...

    #
    @property
    def dtype(self, /) -> np.dtype[_SCT_co]: ...
    @dtype.setter
    def dtype(self, newtype: Never, /) -> None: ...

    #
    @overload
    def __init__(
        self,
        /,
        arg1: _spbase[_SCT_co, _ShapeT_co] | onp.CanArrayND[_SCT_co, _ShapeT_co],
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: _data_matrix[_SCT, tuple[int]],
        /,
        arg1: Sequence[_SCT],
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: _data_matrix[_SCT, tuple[int, int]],
        /,
        arg1: Sequence[onp.CanArrayND[_SCT] | Sequence[_SCT]],
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: _data_matrix[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: _data_matrix[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(self, /, arg1: onp.CanArrayND[_SCT_co], *, maxprint: int | None = None) -> None: ...

    #
    @override
    def __imul__(self, rhs: _ScalarLike, /) -> Self: ...  # type: ignore[override]
    @override
    def __itruediv__(self, rhs: _ScalarLike, /) -> Self: ...  # type: ignore[override]

    # NOTE: The following methods do not convert the scalar type
    def sign(self, /) -> Self: ...
    def ceil(self, /) -> Self: ...
    def floor(self, /) -> Self: ...
    def rint(self, /) -> Self: ...
    def trunc(self, /) -> Self: ...

    #
    @overload
    def sqrt(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def sqrt(self, /) -> Self: ...
    #
    @overload
    def expm1(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def expm1(self, /) -> Self: ...
    #
    @overload
    def log1p(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def log1p(self, /) -> Self: ...

    #
    @overload
    def sin(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def sin(self, /) -> Self: ...
    #
    @overload
    def arcsin(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def arcsin(self, /) -> Self: ...
    #
    @overload
    def sinh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def sinh(self, /) -> Self: ...
    #
    @overload
    def arcsinh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def arcsinh(self, /) -> Self: ...
    #
    @overload
    def tan(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def tan(self, /) -> Self: ...
    #
    @overload
    def arctan(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def arctan(self, /) -> Self: ...
    #
    @overload
    def tanh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def tanh(self, /) -> Self: ...
    #
    @overload
    def arctanh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def arctanh(self, /) -> Self: ...

    #
    @overload
    def deg2rad(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def deg2rad(self, /) -> Self: ...
    #
    @overload
    def rad2deg(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64, _ShapeT_co]: ...
    @overload
    def rad2deg(self, /) -> Self: ...

class _minmax_mixin(Generic[_SCT_co, _ShapeT_co]):
    # NOTE: The following 4 methods have identical signatures
    @overload  # axis: None = ..., out: None = ...
    def max(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> _SCT_co: ...
    @overload  # 1-d, axis: int, out: None = ...
    def max(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> _SCT_co: ...
    @overload  # sparray, axis: int, out: None = ...
    def max(self: sparray, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_array[_SCT_co, tuple[int]]: ...  # type: ignore[misc]
    @overload  # spmatrix, axis: int, out: None = ...
    def max(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_matrix[_SCT_co]: ...  # type: ignore[misc]
    #
    @overload  # axis: None = ..., out: None = ...
    def nanmax(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> _SCT_co: ...
    @overload  # 1-d, axis: int, out: None = ...
    def nanmax(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> _SCT_co: ...
    @overload  # sparray, axis: int, out: None = ...
    def nanmax(  # type: ignore[misc]
        self: sparray,
        /,
        axis: onp.ToInt,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> coo_array[_SCT_co, tuple[int]]: ...
    @overload  # spmatrix, axis: int, out: None = ...
    def nanmax(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_matrix[_SCT_co]: ...  # type: ignore[misc]
    #
    @overload  # axis: None = ..., out: None = ...
    def min(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> _SCT_co: ...
    @overload  # 1-d, axis: int, out: None = ...
    def min(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> _SCT_co: ...
    @overload  # sparray, axis: int, out: None = ...
    def min(self: sparray, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_array[_SCT_co, tuple[int]]: ...  # type: ignore[misc]
    @overload  # spmatrix, axis: int, out: None = ...
    def min(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_matrix[_SCT_co]: ...  # type: ignore[misc]
    #
    @overload  # axis: None = ..., out: None = ...
    def nanmin(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> _SCT_co: ...
    @overload  # 1-d, axis: int, out: None = ...
    def nanmin(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> _SCT_co: ...
    @overload  # sparray, axis: int, out: None = ...
    def nanmin(  # type: ignore[misc]
        self: sparray,
        /,
        axis: onp.ToInt,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> coo_array[_SCT_co, tuple[int]]: ...
    @overload  # spmatrix, axis: int, out: None = ...
    def nanmin(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> coo_matrix[_SCT_co]: ...  # type: ignore[misc]

    # NOTE: The following 2 methods have identical signatures
    @overload  # axis: None = ..., out: None = ...
    def argmax(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> int: ...
    @overload  # 1-d, axis: int, out: None = ...
    def argmax(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> int: ...
    @overload  # sparray, axis: int, out: None = ...
    def argmax(self: sparray, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> onp.Array1D[np.intp]: ...  # type: ignore[misc]
    @overload  # spmatrix, axis: int, out: None = ...
    def argmax(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> onp.Matrix[np.intp]: ...  # type: ignore[misc]

    #
    @overload  # axis: None = ..., out: None = ...
    def argmin(self, /, axis: None = None, out: None = None, *, explicit: bool = False) -> int: ...
    @overload  # 1-d, axis: int, out: None = ...
    def argmin(
        self: _minmax_mixin[Any, tuple[int]],
        /,
        axis: onp.ToInt | None = None,
        out: None = None,
        *,
        explicit: bool = False,
    ) -> int: ...
    @overload  # sparray, axis: int, out: None = ...
    def argmin(self: sparray, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> onp.Array1D[np.intp]: ...  # type: ignore[misc]
    @overload  # spmatrix, axis: int, out: None = ...
    def argmin(self: spmatrix, /, axis: onp.ToInt, out: None = None, *, explicit: bool = False) -> onp.Matrix[np.intp]: ...  # type: ignore[misc]
