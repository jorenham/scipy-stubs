from collections.abc import Iterable, Sequence as Seq
from typing import Any, Final, Literal, Protocol, TypeAlias, TypedDict, TypeVar, overload, type_check_only
from typing_extensions import TypeIs

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from scipy._typing import OrderKACF
from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
)
from scipy.sparse._typing import ToDType

__all__ = [
    "broadcast_shapes",
    "get_sum_dtype",
    "getdata",
    "getdtype",
    "isdense",
    "isintlike",
    "ismatrix",
    "isscalarlike",
    "issequence",
    "isshape",
    "upcast",
]

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=Any)
_DTypeT = TypeVar("_DTypeT", bound=np.dtype[Any])
_SCT = TypeVar("_SCT", bound=np.generic, default=Any)
_IntT = TypeVar("_IntT", bound=np.integer[Any])
_NonIntDTypeT = TypeVar(
    "_NonIntDTypeT",
    bound=np.dtype[np.inexact[Any] | np.flexible | np.datetime64 | np.timedelta64 | np.object_],
)

_SupportedScalar: TypeAlias = np.bool_ | np.integer[Any] | np.float32 | np.float64 | np.longdouble | np.complexfloating[Any, Any]
_ShapeLike: TypeAlias = Iterable[op.CanIndex]
_ScalarLike: TypeAlias = complex | bytes | str | np.generic | onp.Array0D
_SequenceLike: TypeAlias = tuple[_ScalarLike, ...] | list[_ScalarLike] | onp.Array1D
_MatrixLike: TypeAlias = tuple[_SequenceLike, ...] | list[_SequenceLike] | onp.Array2D

_ToArray: TypeAlias = onp.CanArrayND[_SCT, _ShapeT] | onp.SequenceND[_SCT | complex | bytes | str]
_ToArray2D: TypeAlias = onp.CanArrayND[_SCT, _ShapeT] | Seq[Seq[_SCT | complex | bytes | str] | onp.CanArrayND[_SCT]]

@type_check_only
class _ReshapeKwargs(TypedDict, total=False):
    order: Literal["C", "F"]
    copy: bool

@type_check_only
class _SizedIndexIterable(Protocol):
    def __len__(self, /) -> int: ...
    def __iter__(self, /) -> op.CanNext[op.CanIndex]: ...

###

supported_dtypes: Final[list[type[_SupportedScalar]]] = ...

#
# NOTE: Technically any `numpy.generic` could be returned, but we only care about the supported scalar types in `scipy.sparse`.
def upcast(*args: npt.DTypeLike) -> _SupportedScalar: ...
def upcast_char(*args: npt.DTypeLike) -> _SupportedScalar: ...
def upcast_scalar(dtype: npt.DTypeLike, scalar: _ScalarLike) -> np.dtype[_SupportedScalar]: ...

#
def downcast_intp_index(
    arr: onp.Array[_ShapeT, np.bool_ | np.integer[Any] | np.floating[Any] | np.timedelta64 | np.object_],
) -> onp.Array[_ShapeT, np.intp]: ...

#
@overload
def to_native(A: _SCT) -> onp.Array0D[_SCT]: ...
@overload
def to_native(A: onp.Array[_ShapeT, _SCT]) -> onp.Array[_ShapeT, _SCT]: ...
@overload
def to_native(A: onp.HasDType[_DTypeT]) -> np.ndarray[Any, _DTypeT]: ...

#
def getdtype(
    dtype: ToDType[_SCT] | None,
    a: onp.HasDType[np.dtype[_SCT]] | None = None,
    default: ToDType[_SCT] | None = None,
) -> np.dtype[_SCT]: ...

#
@overload
def getdata(obj: _SCT | complex | bytes | str, dtype: ToDType[_SCT] | None = None, copy: bool = False) -> onp.Array0D[_SCT]: ...
@overload
def getdata(obj: _ToArray[_SCT, _ShapeT], dtype: ToDType[_SCT] | None = None, copy: bool = False) -> onp.Array[_ShapeT, _SCT]: ...

#
def get_index_dtype(
    arrays: tuple[onp.ToInt | onp.ToIntND, ...] = (),
    maxval: onp.ToFloat | None = None,
    check_contents: op.CanBool = False,
) -> np.int32 | np.int64: ...

# NOTE: The inline annotations (`(np.dtype) -> np.dtype`) are incorrect.
@overload
def get_sum_dtype(dtype: np.dtype[np.unsignedinteger[Any]]) -> type[np.uint]: ...
@overload
def get_sum_dtype(dtype: np.dtype[np.bool_ | np.signedinteger[Any]]) -> type[np.int_]: ...
@overload
def get_sum_dtype(dtype: _NonIntDTypeT) -> _NonIntDTypeT: ...

#
# NOTE: all arrays implement `__index__` but if it raises this returns `False`, so `TypeIs` can't be used here
def isintlike(x: object) -> TypeIs[op.CanIndex]: ...
def isscalarlike(x: object) -> TypeIs[_ScalarLike]: ...
def isshape(x: _SizedIndexIterable, nonneg: bool = False, *, allow_nd: tuple[int, ...] = (2,)) -> bool: ...
def issequence(t: object) -> TypeIs[_SequenceLike]: ...
def ismatrix(t: object) -> TypeIs[_MatrixLike]: ...
def isdense(x: object) -> TypeIs[onp.Array]: ...

#
def validateaxis(axis: Literal[-2, -1, 0, 1] | bool | np.bool_ | np.integer[Any] | None) -> None: ...
def check_shape(
    args: _ShapeLike | tuple[_ShapeLike, ...],
    current_shape: tuple[int, ...] | None = None,
    *,
    allow_nd: tuple[int, ...] = (2,),
) -> tuple[int, ...]: ...
def check_reshape_kwargs(kwargs: _ReshapeKwargs) -> Literal["C", "F"] | bool: ...

#
def matrix(
    object: _ToArray2D[_SCT],
    dtype: ToDType[_SCT] | type | str | None = None,
    *,
    copy: Literal[0, 1, 2] | bool | None = True,
    order: OrderKACF = "K",
    subok: bool = False,
    ndmin: Literal[0, 1, 2] = 0,
    like: onp.CanArrayFunction | None = None,
) -> onp.Matrix[_SCT]: ...

#
def asmatrix(data: _ToArray2D[_SCT], dtype: ToDType[_SCT] | type | str | None = None) -> onp.Matrix[_SCT]: ...

#
@overload  # BSR/CSC/CSR, dtype: <default>
def safely_cast_index_arrays(
    A: bsr_array | bsr_matrix | csc_array | csc_matrix | csr_array | csr_matrix,
    idx_dtype: ToDType[np.int32] = ...,
    msg: str = "",
) -> tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]]: ...
@overload  # BSR/CSC/CSR, dtype: <known>
def safely_cast_index_arrays(
    A: bsr_array | bsr_matrix | csc_array | csc_matrix | csr_array | csr_matrix,
    idx_dtype: ToDType[_IntT],
    msg: str = "",
) -> tuple[onp.Array1D[_IntT], onp.Array1D[_IntT]]: ...
@overload  # 2d COO, dtype: <default>
def safely_cast_index_arrays(
    A: coo_array[Any, tuple[int, int]] | coo_matrix,
    idx_dtype: ToDType[np.int32] = ...,
    msg: str = "",
) -> tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]]: ...
@overload  # 2d COO, dtype: <known>
def safely_cast_index_arrays(
    A: coo_array[Any, tuple[int, int]] | coo_matrix,
    idx_dtype: ToDType[_IntT],
    msg: str = "",
) -> tuple[onp.Array1D[_IntT], onp.Array1D[_IntT]]: ...
@overload  # nd COO, dtype: <default>
def safely_cast_index_arrays(
    A: coo_array,
    idx_dtype: ToDType[np.int32] = ...,
    msg: str = "",
) -> tuple[onp.Array1D[np.int32], ...]: ...
@overload  # nd COO, dtype: <known>
def safely_cast_index_arrays(
    A: coo_array,
    idx_dtype: ToDType[_IntT],
    msg: str = "",
) -> tuple[onp.Array1D[_IntT], ...]: ...
@overload  # DIA, dtype: <default>
def safely_cast_index_arrays(
    A: dia_array | dia_matrix,
    idx_dtype: ToDType[np.int32] = ...,
    msg: str = "",
) -> onp.Array1D[np.int32]: ...
@overload  # DIA, dtype: <known>
def safely_cast_index_arrays(
    A: dia_array | dia_matrix,
    idx_dtype: ToDType[_IntT],
    msg: str = "",
) -> onp.Array1D[_IntT]: ...

#
def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]: ...
