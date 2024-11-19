from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_LapackDriver: TypeAlias = Literal["gesdd", "gesvd"]

_SCT = TypeVar("_SCT", bound=np.generic)
_DT = TypeVar("_DT", bound=np.dtype[np.generic])

_Float1D: TypeAlias = onp.Array1D[np.float32 | np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float32 | np.float64]
_Complex2D: TypeAlias = onp.Array2D[np.complex64 | np.complex128]

@overload
def svd(
    a: onp.ToFloat2D,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Float2D, _Float1D, _Float2D]: ...
@overload
def svd(
    a: onp.AnyComplexFloatingArray,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Complex2D, _Float1D, _Complex2D]: ...
@overload
def svd(
    a: onp.ToComplex2D,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[_Float2D, _Float1D, _Float2D] | tuple[_Complex2D, _Float1D, _Complex2D]: ...
@overload
def svd(
    a: onp.ToComplex2D,
    full_matrices: bool,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Float1D: ...
@overload
def svd(
    a: onp.ToComplex2D,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Float1D: ...

#
def svdvals(a: onp.ToComplex2D, overwrite_a: bool = False, check_finite: bool = True) -> _Float1D: ...

# beware the overlapping overloads for bool <: int (<: float)
@overload
def diagsvd(s: onp.CanArray[tuple[int, ...], _DT], M: op.CanIndex, N: op.CanIndex) -> np.ndarray[tuple[int, int], _DT]: ...
@overload
def diagsvd(s: Sequence[_SCT], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[_SCT]: ...
@overload
def diagsvd(s: Sequence[bool], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_]: ...
@overload
def diagsvd(s: Sequence[int], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_ | np.int_]: ...
@overload
def diagsvd(s: Sequence[float], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_ | np.int_ | np.float64]: ...

#
@overload
def orth(A: onp.ToFloat2D, rcond: float | np.floating[Any] | None = None) -> _Float2D: ...
@overload
def orth(A: onp.AnyComplexFloatingArray, rcond: float | np.floating[Any] | None = None) -> _Complex2D: ...
@overload
def orth(A: onp.ToComplex2D, rcond: float | np.floating[Any] | None = None) -> _Float2D | _Complex2D: ...

#
@overload
def null_space(A: onp.ToFloat2D, rcond: float | np.floating[Any] | None = None) -> _Float2D: ...
@overload
def null_space(A: onp.AnyComplexFloatingArray, rcond: float | np.floating[Any] | None = None) -> _Complex2D: ...
@overload
def null_space(A: onp.ToComplex2D, rcond: float | np.floating[Any] | None = None) -> _Float2D | _Complex2D: ...

#
def subspace_angles(A: onp.ToComplex2D, B: onp.ToComplex2D) -> _Float1D: ...
