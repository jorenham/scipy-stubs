# NOTE: Adding `@override` to `@overload`ed methods will crash stubtest (basedmypy 1.13.0)
# mypy: disable-error-code="misc, override, explicit-override"
# pyright: reportIncompatibleMethodOverride=false

from collections.abc import Iterable, Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Never, Self, TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix
from ._typing import Numeric, ToShapeMin1D

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

###

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

_1D: TypeAlias = tuple[int]  # noqa: PYI042
_2D: TypeAlias = tuple[int, int]  # noqa: PYI042

_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToKey1D: TypeAlias = onp.ToJustInt | tuple[onp.ToJustInt]
_ToKey2D: TypeAlias = tuple[onp.ToJustInt, onp.ToJustInt]

_ToKeys1D: TypeAlias = Iterable[_ToKey1D]
_ToKeys2D: TypeAlias = Iterable[_ToKey2D]

###

class _dok_base(
    _spbase[_SCT, _ShapeT_co],
    IndexMixin[_SCT, _ShapeT_co],
    dict[tuple[int] | tuple[int, int], _SCT],
    Generic[_SCT, _ShapeT_co],
):
    dtype: np.dtype[_SCT]

    @property
    @override
    def format(self, /) -> Literal["dok"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def shape(self, /) -> _ShapeT_co: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT],
        shape: ToShapeMin1D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _dok_base[np.float64],
        /,
        arg1: ToShapeMin1D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _dok_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _dok_base[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _dok_base[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _dok_base[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShapeMin1D | None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShapeMin1D | None = None,
        *,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @override
    def todok(self, /, copy: bool = False) -> Self: ...

    #
    @override
    def __len__(self, /) -> int: ...

    #
    @overload
    def __delitem__(self: _dok_base[Any, _2D], key: _ToKey2D, /) -> None: ...
    @overload
    def __delitem__(self: _dok_base[Any, _1D], key: _ToKey1D, /) -> None: ...
    @overload
    def __delitem__(self, key: _ToKey1D | _ToKey2D, /) -> None: ...

    #
    @override
    def __or__(self, other: Never, /) -> Never: ...
    @override
    def __ror__(self, other: Never, /) -> Never: ...
    @override
    def __ior__(self, other: Never, /) -> Never: ...  # noqa: PYI034

    #
    @overload
    def count_nonzero(self, /, axis: None = None) -> int: ...
    @overload
    def count_nonzero(self, /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...

    #
    @override
    def update(self, /, val: Never) -> Never: ...

    #
    @overload
    def setdefault(self: _dok_base[Any, _2D], key: _ToKey2D, default: _T, /) -> _SCT | _T: ...
    @overload
    def setdefault(self: _dok_base[Any, _2D], key: _ToKey2D, default: None = None, /) -> _SCT | None: ...
    @overload
    def setdefault(self: _dok_base[Any, _1D], key: _ToKey1D, default: _T, /) -> _SCT | _T: ...
    @overload
    def setdefault(self: _dok_base[Any, _1D], key: _ToKey1D, default: None = None, /) -> _SCT | None: ...
    @overload
    def setdefault(self, key: _ToKey1D | _ToKey2D, default: _T, /) -> _SCT | _T: ...
    @overload
    def setdefault(self, key: _ToKey1D | _ToKey2D, default: None = None, /) -> _SCT | None: ...

    #
    @overload
    def get(self: _dok_base[Any, _2D], /, key: _ToKey2D, default: _T) -> _SCT | _T: ...
    @overload
    def get(self: _dok_base[Any, _2D], /, key: _ToKey2D, default: float = 0.0) -> _SCT | float: ...
    @overload
    def get(self: _dok_base[Any, _1D], /, key: _ToKey1D, default: _T) -> _SCT | _T: ...
    @overload
    def get(self: _dok_base[Any, _1D], /, key: _ToKey1D, default: float = 0.0) -> _SCT | float: ...
    @overload
    def get(self, /, key: _ToKey1D | _ToKey2D, default: _T) -> _SCT | _T: ...
    @overload
    def get(self, /, key: _ToKey1D | _ToKey2D, default: float = 0.0) -> _SCT | float: ...

    #
    def conjtransp(self, /) -> Self: ...

    #
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[_SCT, _2D]], iterable: _ToKeys2D, v: _SCT, /) -> _dok_base[_SCT, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[_SCT, _1D]], iterable: _ToKeys1D, v: _SCT, /) -> _dok_base[_SCT, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool_, _2D]], iterable: _ToKeys2D, v: onp.ToBool, /) -> _dok_base[np.bool_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool_, _1D]], iterable: _ToKeys1D, v: onp.ToBool, /) -> _dok_base[np.bool_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.int_, _2D]], iterable: _ToKeys2D, v: op.JustInt = 1, /) -> _dok_base[np.int_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.int_, _1D]], iterable: _ToKeys1D, v: op.JustInt = 1, /) -> _dok_base[np.int_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[_dok_base[np.float64, _2D]],
        iterable: _ToKeys2D,
        v: op.JustFloat,
        /,
    ) -> _dok_base[np.float64, _2D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[_dok_base[np.float64, _1D]],
        iterable: _ToKeys1D,
        v: op.JustFloat,
        /,
    ) -> _dok_base[np.float64, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[_dok_base[np.complex128, _2D]],
        iterable: _ToKeys2D,
        v: op.JustComplex,
        /,
    ) -> _dok_base[np.complex128, _2D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[_dok_base[np.complex128, _1D]],
        iterable: _ToKeys1D,
        v: op.JustComplex,
        /,
    ) -> _dok_base[np.complex128, _1D]: ...

#
class dok_array(_dok_base[_SCT, _ShapeT_co], sparray[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    # NOTE: This horrible code duplication is required due to the lack of higher-kinded typing (HKT) support.
    # https://github.com/python/typing/issues/548
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[_SCT, _2D]], iterable: _ToKeys2D, v: _SCT, /) -> dok_array[_SCT, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[_SCT, _1D]], iterable: _ToKeys1D, v: _SCT, /) -> dok_array[_SCT, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool_, _2D]], iterable: _ToKeys2D, v: onp.ToBool, /) -> dok_array[np.bool_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool_, _1D]], iterable: _ToKeys1D, v: onp.ToBool, /) -> dok_array[np.bool_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _2D]], iterable: _ToKeys2D, v: op.JustInt = 1, /) -> dok_array[np.int_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _1D]], iterable: _ToKeys1D, v: op.JustInt = 1, /) -> dok_array[np.int_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.float64, _2D]],
        iterable: _ToKeys2D,
        v: op.JustFloat,
        /,
    ) -> dok_array[np.float64, _2D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.float64, _1D]],
        iterable: _ToKeys1D,
        v: op.JustFloat,
        /,
    ) -> dok_array[np.float64, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.complex128, _2D]],
        iterable: _ToKeys2D,
        v: op.JustComplex,
        /,
    ) -> dok_array[np.complex128, _2D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.complex128, _1D]],
        iterable: _ToKeys1D,
        v: op.JustComplex,
        /,
    ) -> dok_array[np.complex128, _1D]: ...

#
class dok_matrix(_dok_base[_SCT, _2D], spmatrix[_SCT], Generic[_SCT]):
    @override
    def get(self, /, key: _ToKey2D, default: onp.ToComplex = 0.0) -> _SCT: ...
    @override
    def setdefault(self, key: _ToKey2D, default: onp.ToComplex | None = None, /) -> _SCT: ...

    #
    @overload
    @classmethod
    def fromkeys(cls, iterable: _ToKeys2D, v: _SCT, /) -> Self: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.bool_]], iterable: _ToKeys2D, v: onp.ToBool, /) -> dok_matrix[np.bool_]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.int_]], iterable: _ToKeys2D, v: op.JustInt = 1, /) -> dok_matrix[np.int_]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.float64]], iterable: _ToKeys2D, v: op.JustFloat, /) -> dok_matrix[np.float64]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_matrix[np.complex128]],
        iterable: _ToKeys2D,
        v: op.JustComplex,
        /,
    ) -> dok_matrix[np.complex128]: ...

#
def isspmatrix_dok(x: object) -> TypeIs[dok_matrix]: ...
