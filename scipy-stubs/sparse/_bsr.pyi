from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase, sparray
from ._compressed import _cs_matrix
from ._data import _minmax_mixin
from ._matrix import spmatrix
from ._typing import Int, Scalar, ToShape2D

__all__ = ["bsr_array", "bsr_matrix", "isspmatrix_bsr"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)

_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToData2: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int]]
_ToData3: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Int], onp.ArrayND[Int]]
_ToData: TypeAlias = _ToData2[_SCT] | _ToData3[_SCT]

###

class _bsr_base(_cs_matrix[_SCT, tuple[int, int]], _minmax_mixin[_SCT, tuple[int, int]], Generic[_SCT]):
    data: onp.Array3D[_SCT]

    @property
    @override
    def format(self, /) -> Literal["bsr"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...
    @property
    def blocksize(self, /) -> tuple[int, int]: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _bsr_base[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _bsr_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _bsr_base[np.int_],
        /,
        arg1: _ToMatrixPy[opt.JustInt],
        shape: ToShape2D | None = None,
        dtype: type[opt.JustInt] | onp.AnyIntPDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _bsr_base[np.float64],
        /,
        arg1: _ToMatrixPy[opt.Just[float]],
        shape: ToShape2D | None = None,
        dtype: type[opt.Just[float]] | onp.AnyFloat64DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _bsr_base[np.complex128],
        /,
        arg1: _ToMatrixPy[opt.Just[complex]],
        shape: ToShape2D | None = None,
        dtype: type[opt.Just[complex]] | onp.AnyComplex128DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None,
        dtype: _ToDType[_SCT],
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
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
        blocksize: tuple[int, int] | None = None,
    ) -> None: ...

class bsr_array(_bsr_base[_SCT], sparray, Generic[_SCT]): ...
class bsr_matrix(_bsr_base[_SCT], spmatrix[_SCT], Generic[_SCT]): ...  # type: ignore[misc]

def isspmatrix_bsr(x: object) -> TypeIs[bsr_matrix]: ...
