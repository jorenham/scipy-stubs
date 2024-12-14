# mypy: disable-error-code="misc, override"
# pyright: reportIncompatibleMethodOverride=false

from collections.abc import Iterable, Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Never, Self, TypeIs, TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix
from ._typing import Scalar, ToShape

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], covariant=True, default=tuple[int] | tuple[int, int])

_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_Key: TypeAlias = tuple[onp.ToJustInt] | tuple[onp.ToJustInt, onp.ToJustInt]

###

class _dok_base(_spbase[_SCT, _ShapeT_co], IndexMixin[_SCT, _ShapeT_co], dict[_Key, _SCT], Generic[_SCT, _ShapeT_co]):
    dtype: np.dtype[_SCT]

    @property
    @override
    def format(self, /) -> Literal["dok"]: ...
    #
    @property
    @override
    def shape(self, /) -> _ShapeT_co: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT],
        shape: ToShape | None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _dok_base[np.float64],
        /,
        arg1: ToShape,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _dok_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _dok_base[np.int_],
        /,
        arg1: _ToMatrixPy[opt.JustInt],
        shape: ToShape | None = None,
        dtype: type[opt.JustInt] | onp.AnyIntPDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _dok_base[np.float64],
        /,
        arg1: _ToMatrixPy[opt.Just[float]],
        shape: ToShape | None = None,
        dtype: type[opt.Just[float]] | onp.AnyFloat64DType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _dok_base[np.complex128],
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
    @override
    def __delitem__(self, key: onp.ToJustInt, /) -> None: ...

    #
    @override
    def __or__(self, other: Never, /) -> Never: ...
    @override
    def __ror__(self, other: Never, /) -> Never: ...
    @override
    def __ior__(self, other: Never, /) -> Never: ...  # noqa: PYI034
    @override
    def update(self, /, val: Never) -> Never: ...

    # TODO(jorenham)
    @override
    def get(self, key: onp.ToJustInt | _Key, /, default: onp.ToComplex = 0.0) -> _SCT: ...
    @override
    def setdefault(self, key: onp.ToJustInt | _Key, default: onp.ToComplex | None = None, /) -> _SCT: ...
    @classmethod
    @override
    def fromkeys(cls, iterable: Iterable[_Key], value: int = 1, /) -> Self: ...

    #
    def conjtransp(self, /) -> Self: ...

class dok_array(_dok_base[_SCT, _ShapeT_co], sparray, Generic[_SCT, _ShapeT_co]): ...

class dok_matrix(_dok_base[_SCT, tuple[int, int]], spmatrix[_SCT], Generic[_SCT]):
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...

    #
    @override
    def get(self, key: tuple[onp.ToJustInt, onp.ToJustInt], /, default: onp.ToComplex = 0.0) -> _SCT: ...
    @override
    def setdefault(self, key: tuple[onp.ToJustInt, onp.ToJustInt], default: onp.ToComplex | None = None, /) -> _SCT: ...

def isspmatrix_dok(x: object) -> TypeIs[dok_matrix]: ...
