import abc
from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Self, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase
from ._data import _data_matrix, _minmax_mixin
from ._index import IndexMixin
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

__all__: list[str] = []

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_SCT0 = TypeVar("_SCT0", bound=Scalar)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToData2B: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int]]  # bsr
_ToData2C: TypeAlias = tuple[onp.ArrayND[_SCT], tuple[onp.ArrayND[Int], onp.ArrayND[Int]]]  # csc, csr
_ToData2: TypeAlias = _ToData2B[_SCT] | _ToData2C[_SCT]
_ToData3: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int], onp.ArrayND[Int]]
_ToData: TypeAlias = _ToData2[_SCT] | _ToData3[_SCT]

###

class _cs_matrix(
    _data_matrix[_SCT, _ShapeT_co],
    _minmax_mixin[_SCT, _ShapeT_co],
    IndexMixin[_SCT, _ShapeT_co],
    Generic[_SCT, _ShapeT_co],
):
    data: onp.Array[Any, _SCT]  # the `Any` shape is needed for `numpy<2.1`
    indices: Index1D
    indptr: Index1D

    @property
    @override
    @abc.abstractmethod
    def format(self, /) -> Literal["bsr", "csc", "csr"]: ...

    #
    @property
    def has_canonical_format(self, /) -> bool: ...
    @has_canonical_format.setter
    def has_canonical_format(self, val: bool, /) -> None: ...
    #
    @property
    def has_sorted_indices(self, /) -> bool: ...
    @has_sorted_indices.setter
    def has_sorted_indices(self, val: bool, /) -> None: ...

    #
    @overload  # sparse or dense (know dtype & shape), dtype: None
    def __init__(
        self,
        /,
        arg1: _spbase[_SCT, _ShapeT_co] | onp.CanArrayND[_SCT, _ShapeT_co],
        shape: _ShapeT_co | None = None,
        dtype: ToDType[_SCT] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d array-like (know dtype), dtype: None
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int]],
        /,
        arg1: Sequence[_SCT0],
        shape: ToShape1D | None = None,
        dtype: ToDType[_SCT0] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d array-like (know dtype), dtype: None
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[_SCT0]] | Sequence[onp.CanArrayND[_SCT0]],  # assumes max. 2-d
        shape: ToShape2D | None = None,
        dtype: ToDType[_SCT0] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape | None = None,
        dtype: ToDType[_SCT] | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: None
    def __init__(
        self: _cs_matrix[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        shape: ToShape1D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _cs_matrix[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        shape: ToShape2D | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d array-like bool, dtype: type[bool] | None
    def __init__(
        self: _cs_matrix[np.bool_, tuple[int]],
        /,
        arg1: Sequence[bool],
        shape: ToShape1D | None = None,
        dtype: ToDTypeBool | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d array-like bool, dtype: type[bool] | None
    def __init__(
        self: _cs_matrix[np.bool_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: ToShape2D | None = None,
        dtype: ToDTypeBool | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d array-like ~int, dtype: type[int] | None
    def __init__(
        self: _cs_matrix[np.int_, tuple[int]],
        /,
        arg1: Sequence[opt.JustInt],
        shape: ToShape | None = None,
        dtype: ToDTypeInt | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d array-like ~int, dtype: type[int] | None
    def __init__(
        self: _cs_matrix[np.int_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.JustInt]],
        shape: ToShape | None = None,
        dtype: ToDTypeInt | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d array-like ~float, dtype: type[float] | None
    def __init__(
        self: _cs_matrix[np.float64, tuple[int]],
        /,
        arg1: Sequence[opt.Just[float]],
        shape: ToShape | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: type[float] | None
    def __init__(
        self: _cs_matrix[np.float64, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.Just[float]]],
        shape: ToShape | None = None,
        dtype: ToDTypeFloat | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-d array-like ~complex, dtype: type[complex] | None
    def __init__(
        self: _cs_matrix[np.complex128, tuple[int]],
        /,
        arg1: Sequence[opt.Just[complex]],
        shape: ToShape | None = None,
        dtype: ToDTypeComplex | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: type[complex] | None
    def __init__(
        self: _cs_matrix[np.complex128, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[opt.Just[complex]]],
        shape: ToShape | None = None,
        dtype: ToDTypeComplex | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (keyword)
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None = None,
        *,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None,
        dtype: ToDType[_SCT0],
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__(
        self: _cs_matrix[_SCT0, tuple[int, int]],
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
    @override
    def count_nonzero(self, /, axis: None = None) -> int: ...

    #
    def sorted_indices(self, /) -> Self: ...
    def sort_indices(self, /) -> None: ...

    #
    def check_format(self, /, full_check: bool = True) -> None: ...
    def eliminate_zeros(self, /) -> None: ...
    def sum_duplicates(self, /) -> None: ...
    def prune(self, /) -> None: ...
