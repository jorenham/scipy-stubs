from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from ._base import _spbase, sparray
from ._data import _data_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Integer, Numeric, ToShape2D

__all__ = ["dia_array", "dia_matrix", "isspmatrix_dia"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)

_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]
_ToData: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[Integer]]

###

class _dia_base(_data_matrix[_SCT, tuple[int, int]], Generic[_SCT]):
    data: onp.Array2D[_SCT]
    offsets: Index1D

    @property
    @override
    def format(self, /) -> Literal["dia"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: _dia_base[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: _dia_base[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: _dia_base[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: _dia_base[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: _dia_base[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShape2D | None = None,
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
        shape: ToShape2D | None,
        dtype: onp.ToDType[_SCT],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None = None,
        *,
        dtype: onp.ToDType[_SCT],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...

class dia_array(_dia_base[_SCT], sparray[_SCT, tuple[int, int]], Generic[_SCT]): ...
class dia_matrix(_dia_base[_SCT], spmatrix[_SCT], Generic[_SCT]): ...  # type: ignore[misc]

def isspmatrix_dia(x: object) -> TypeIs[dia_matrix]: ...
