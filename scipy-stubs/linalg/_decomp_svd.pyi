from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onpt

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_LapackDriver: TypeAlias = Literal["gesdd", "gesvd"]

_SCT = TypeVar("_SCT", bound=np.generic)
_DT = TypeVar("_DT", bound=np.dtype[np.generic])

_Array_1d: TypeAlias = onpt.Array[tuple[int], _SCT]
_Array_2d: TypeAlias = onpt.Array[tuple[int, int], _SCT]

_ArrayLike_2d_f: TypeAlias = Sequence[Sequence[float]] | onpt.AnyFloatingArray
_ArrayLike_2d_c: TypeAlias = onpt.AnyComplexFloatingArray  # because `float <: complex` (type-check only), complex is excluded
_ArrayLike_2d_fc: TypeAlias = Sequence[Sequence[complex]] | onpt.AnyNumberArray

_Array_1d_f: TypeAlias = _Array_1d[np.float32 | np.float64]
_Array_2d_f: TypeAlias = _Array_2d[np.float32 | np.float64]
_Array_2d_c: TypeAlias = _Array_2d[np.complex64 | np.complex128]

@overload
def svd(
    a: _ArrayLike_2d_f,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Array_2d_f, _Array_1d_f, _Array_2d_f]: ...
@overload
def svd(
    a: _ArrayLike_2d_c,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Array_2d_c, _Array_1d_f, _Array_2d_c]: ...
@overload
def svd(
    a: _ArrayLike_2d_fc,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Array_2d_f, _Array_1d_f, _Array_2d_f] | tuple[_Array_2d_c, _Array_1d_f, _Array_2d_c]: ...
@overload
def svd(
    a: _ArrayLike_2d_fc,
    full_matrices: bool,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Array_1d_f: ...
@overload
def svd(
    a: _ArrayLike_2d_fc,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Array_1d_f: ...

#
def svdvals(a: _ArrayLike_2d_fc, overwrite_a: bool = False, check_finite: bool = True) -> _Array_1d_f: ...

# beware the overlapping overloads for bool <: int (<: float)
@overload
def diagsvd(s: onpt.CanArray[tuple[int, ...], _DT], M: op.CanIndex, N: op.CanIndex) -> np.ndarray[tuple[int, int], _DT]: ...
@overload
def diagsvd(s: Sequence[_SCT], M: op.CanIndex, N: op.CanIndex) -> _Array_2d[_SCT]: ...
@overload
def diagsvd(s: Sequence[bool], M: op.CanIndex, N: op.CanIndex) -> _Array_2d[np.bool_]: ...
@overload
def diagsvd(s: Sequence[int], M: op.CanIndex, N: op.CanIndex) -> _Array_2d[np.bool_ | np.int_]: ...
@overload
def diagsvd(s: Sequence[float], M: op.CanIndex, N: op.CanIndex) -> _Array_2d[np.bool_ | np.int_ | np.float64]: ...

#
@overload
def orth(A: _ArrayLike_2d_f, rcond: float | np.floating[Any] | None = None) -> _Array_2d_f: ...
@overload
def orth(A: _ArrayLike_2d_c, rcond: float | np.floating[Any] | None = None) -> _Array_2d_c: ...
@overload
def orth(A: _ArrayLike_2d_fc, rcond: float | np.floating[Any] | None = None) -> _Array_2d_f | _Array_2d_c: ...

#
@overload
def null_space(A: _ArrayLike_2d_f, rcond: float | np.floating[Any] | None = None) -> _Array_2d_f: ...
@overload
def null_space(A: _ArrayLike_2d_c, rcond: float | np.floating[Any] | None = None) -> _Array_2d_c: ...
@overload
def null_space(A: _ArrayLike_2d_fc, rcond: float | np.floating[Any] | None = None) -> _Array_2d_f | _Array_2d_c: ...

#
def subspace_angles(A: _ArrayLike_2d_fc, B: _ArrayLike_2d_fc) -> _Array_1d_f: ...
