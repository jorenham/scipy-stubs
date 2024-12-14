# mypy: disable-error-code="explicit-override"

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._matrix import spmatrix
from ._typing import (
    Index1D,
    Int,
    Scalar,
    ToDType,
    ToDTypeBool,
    ToDTypeComplex,
    ToDTypeFloat,
    ToDTypeInt,
    ToShape,
    ToShape1D,
    ToShape2D,
)

__all__ = ["coo_array", "coo_matrix", "isspmatrix_coo"]

_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_SCT0 = TypeVar("_SCT0", bound=Scalar)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

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
        arg1: _spbase[_SCT, _ShapeT_co],
        shape: _ShapeT_co | None = None,
        dtype: ToDType[_SCT] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToData[_SCT],
        shape: _ShapeT_co | None = None,
        dtype: ToDType[_SCT] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: None
    def __init__(  # type: ignore[misc]
        self: coo_array[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        shape: ToShape1D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _coo_base[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        shape: ToShape2D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # vector-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _coo_base[np.bool_, tuple[int]],
        /,
        arg1: Sequence[bool],
        shape: ToShape1D | None = None,
        dtype: ToDTypeBool | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _coo_base[np.bool_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeBool | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # vector-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _coo_base[np.int_, tuple[int]],
        /,
        arg1: Sequence[opt.JustInt],
        shape: ToShape1D | None = None,
        dtype: ToDTypeInt | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _coo_base[np.int_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.JustInt]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeInt | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # vectir-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _coo_base[np.float64, tuple[int]],
        /,
        arg1: Sequence[opt.Just[float]],
        shape: ToShape1D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _coo_base[np.float64, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.Just[float]]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _coo_base[np.complex128, tuple[int]],
        /,
        arg1: Sequence[opt.Just[complex]],
        shape: ToShape1D | None = None,
        dtype: ToDTypeComplex | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _coo_base[np.complex128, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.Just[complex]]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeComplex | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__(
        self: _coo_base[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (keyword)
    def __init__(
        self: _coo_base[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None = None,
        *,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__(
        self: _coo_base[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__(
        self: _coo_base[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None = None,
        *,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # shape: known
    def __init__(
        self,
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: ToShape | None = None,
        dtype: npt.DTypeLike | None = ...,
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
