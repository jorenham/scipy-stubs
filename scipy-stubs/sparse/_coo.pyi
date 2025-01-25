# mypy: disable-error-code="explicit-override"

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from ._base import _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._matrix import spmatrix
from ._typing import Index1D, Integer, Numeric, ToShape1D, ToShape2D, ToShapeMin1D, ToShapeMin3D

__all__ = ["coo_array", "coo_matrix", "isspmatrix_coo"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_SCT0 = TypeVar("_SCT0", bound=Numeric)
_ShapeT_co = TypeVar("_ShapeT_co", bound=onp.AtLeast1D, default=onp.AtLeast1D, covariant=True)

_ToData: TypeAlias = tuple[onp.ArrayND[_SCT0], tuple[onp.ArrayND[Integer]] | tuple[onp.ArrayND[Integer], onp.ArrayND[Integer]]]
_ToDense: TypeAlias = onp.ArrayND[_SCT0] | onp.SequenceND[onp.ArrayND[_SCT0]] | onp.SequenceND[_SCT0]

_ScalarOrDense: TypeAlias = onp.ArrayND[_SCT0] | _SCT0
_JustND: TypeAlias = onp.SequenceND[op.Just[_T]]

_SubInt: TypeAlias = np.bool_ | np.int8 | np.int16 | np.int32 | np.intp | np.int_ | np.uint8 | np.uint16
_SubFloat: TypeAlias = np.bool_ | Integer | np.float32 | np.float64
_SubComplex: TypeAlias = _SubFloat | np.complex64 | np.complex128
_SupComplex: TypeAlias = np.complex128 | np.clongdouble
_SupFloat: TypeAlias = np.float64 | np.longdouble | _SupComplex
_SupInt: TypeAlias = np.int_ | np.int64 | np.uint32 | np.uintp | np.uint | np.uint64 | _SupFloat

_Axes: TypeAlias = int | tuple[Sequence[int], Sequence[int]]

_SupComplexT = TypeVar("_SupComplexT", bound=_SupComplex)
_SupFloatT = TypeVar("_SupFloatT", bound=_SupFloat)
_SupIntT = TypeVar("_SupIntT", bound=_SupInt)

###

class _coo_base(_data_matrix[_SCT, _ShapeT_co], _minmax_mixin[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    data: onp.Array1D[_SCT]
    coords: tuple[Index1D, ...]  # len(coords) == ndim
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
    def row(self, row: onp.ToInt1D, /) -> None: ...
    #
    @property
    def col(self, /) -> Index1D: ...
    @col.setter
    def col(self, col: onp.ToInt1D, /) -> None: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _spbase[_SCT, _ShapeT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_SCT] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToData[_SCT],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_SCT] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: None
    def __init__(  # type: ignore[misc]
        self: coo_array[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        shape: ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _coo_base[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: None
    def __init__(
        self: _coo_base[np.float64, onp.AtLeast3D],
        /,
        arg1: ToShapeMin3D,
        shape: ToShapeMin3D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vector-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _coo_base[np.bool_, tuple[int]],
        /,
        arg1: Sequence[bool],
        shape: ToShape1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _coo_base[np.bool_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vector-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _coo_base[np.int_, tuple[int]],
        /,
        arg1: Sequence[op.JustInt],
        shape: ToShape1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _coo_base[np.int_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustInt]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vectir-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _coo_base[np.float64, tuple[int]],
        /,
        arg1: Sequence[op.JustFloat],
        shape: ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _coo_base[np.float64, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustFloat]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _coo_base[np.complex128, tuple[int]],
        /,
        arg1: Sequence[op.JustComplex],
        shape: ToShape1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _coo_base[np.complex128, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustComplex]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__(
        self: _coo_base[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (keyword)
    def __init__(
        self: _coo_base[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None = None,
        *,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__(
        self: _coo_base[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__(
        self: _coo_base[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None = None,
        *,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-D, dtype: <known> (positional)
    def __init__(
        self: _coo_base[_SCT0, onp.AtLeast3D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShapeMin3D | None,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-D, dtype: <known> (keyword)
    def __init__(
        self: _coo_base[_SCT0, onp.AtLeast3D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShapeMin3D | None = None,
        *,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # shape: known
    def __init__(
        self,
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: ToShapeMin1D | None = None,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    def sum_duplicates(self, /) -> None: ...
    def eliminate_zeros(self, /) -> None: ...

    # NOTE: all combinations (self dtype, other dtype, self shape, other shape, self: array|matrix, other dense|sparse, axes)
    #   would  result in more overloads than that mypy has bugs (i.e. >1_200).
    # NOTE: due to a bug in `axes`, only `int` can be used at the moment (passing a 2-tuple or 2-list raises `TypeError`)
    @overload
    def tensordot(self, /, other: _spbase[_SCT0], axes: _Axes = 2) -> _SCT | _SCT0 | coo_array[_SCT | _SCT0]: ...
    @overload
    def tensordot(self, /, other: _ToDense[_SCT0], axes: _Axes = 2) -> _ScalarOrDense[_SCT | _SCT0]: ...
    @overload
    def tensordot(self, /, other: onp.SequenceND[bool], axes: _Axes = 2) -> _ScalarOrDense[_SCT]: ...
    @overload
    def tensordot(self: _spbase[_SubInt], /, other: _JustND[int], axes: _Axes = 2) -> _ScalarOrDense[np.int_]: ...
    @overload
    def tensordot(self: _spbase[_SubFloat], /, other: _JustND[float], axes: _Axes = 2) -> _ScalarOrDense[np.float64]: ...
    @overload
    def tensordot(self: _spbase[_SubComplex], /, other: _JustND[complex], axes: _Axes = 2) -> _ScalarOrDense[np.complex128]: ...
    @overload
    def tensordot(self: _spbase[_SupComplexT], /, other: _JustND[complex], axes: _Axes = 2) -> _ScalarOrDense[_SupComplexT]: ...
    @overload
    def tensordot(self: _spbase[_SupFloatT], /, other: _JustND[float], axes: _Axes = 2) -> _ScalarOrDense[_SupFloatT]: ...
    @overload
    def tensordot(self: _spbase[_SupIntT], /, other: _JustND[int], axes: _Axes = 2) -> _ScalarOrDense[_SupIntT]: ...

class coo_array(_coo_base[_SCT, _ShapeT_co], sparray[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]): ...

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
