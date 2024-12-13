import abc
from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Self, TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase
from ._data import _data_matrix, _minmax_mixin
from ._index import IndexMixin
from ._typing import Index1D, Int, Scalar, ToShape2D

__all__: list[str] = []

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)

_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToData2B: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int]]  # bsr
_ToData2C: TypeAlias = tuple[onp.ArrayND[_SCT], tuple[onp.ArrayND[Int], onp.ArrayND[Int]]]  # csc, csr
_ToData2: TypeAlias = _ToData2B[_SCT] | _ToData2C[_SCT]
_ToData3: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int], onp.ArrayND[Int]]
_ToData: TypeAlias = _ToData2[_SCT] | _ToData3[_SCT]

###

class _cs_matrix(_data_matrix[_SCT], _minmax_mixin[_SCT], IndexMixin[_SCT], Generic[_SCT]):
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
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _cs_matrix[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _cs_matrix[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _cs_matrix[np.int_],
        /,
        arg1: _ToMatrixPy[opt.JustInt],
        shape: ToShape2D | None = None,
        dtype: type[opt.JustInt] | onp.AnyIntPDType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _cs_matrix[np.float64],
        /,
        arg1: _ToMatrixPy[opt.Just[float]],
        shape: ToShape2D | None = None,
        dtype: type[opt.Just[float]] | onp.AnyFloat64DType | None = None,
        copy: bool = False,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _cs_matrix[np.complex128],
        /,
        arg1: _ToMatrixPy[opt.Just[complex]],
        shape: ToShape2D | None = None,
        dtype: type[opt.Just[complex]] | onp.AnyComplex128DType | None = None,
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
    def count_nonzero(self, /, axis: None = None) -> int: ...

    #
    def sorted_indices(self, /) -> Self: ...
    def sort_indices(self, /) -> None: ...

    #
    def check_format(self, /, full_check: bool = True) -> None: ...
    def eliminate_zeros(self, /) -> None: ...
    def sum_duplicates(self, /) -> None: ...
    def prune(self, /) -> None: ...
