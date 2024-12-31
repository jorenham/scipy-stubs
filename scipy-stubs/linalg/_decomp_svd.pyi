from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_Float: TypeAlias = np.float32 | np.float64
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]

_Complex: TypeAlias = np.complex64 | np.complex128
_Complex2D: TypeAlias = onp.Array2D[_Complex]
_Inexact2D: TypeAlias = onp.Array2D[_Float | _Complex]

_LapackDriver: TypeAlias = Literal["gesdd", "gesvd"]

_FloatSVD: TypeAlias = tuple[_Float2D, _Float1D, _Float2D]
_ComplexSVD: TypeAlias = tuple[_Complex2D, _Float1D, _Complex2D]

_RealT = TypeVar("_RealT", bound=np.bool_ | np.integer[Any] | np.floating[Any])
_InexactT = TypeVar("_InexactT", bound=_Float | _Complex)

###

@overload
def svd(
    a: onp.ToFloat2D,
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatSVD: ...
@overload
def svd(
    a: onp.ToComplex2D,
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatSVD | _ComplexSVD: ...
@overload  # complex, compute_uv: {False}
def svd(
    a: onp.ToComplex2D,
    full_matrices: onp.ToBool,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Float1D: ...
@overload  # complex, *, compute_uv: {False}
def svd(
    a: onp.ToComplex2D,
    full_matrices: onp.ToBool = True,
    *,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Float1D: ...

#
def svdvals(a: onp.ToComplex2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float1D: ...

# beware the overlapping overloads for bool <: int (<: float)
@overload
def diagsvd(s: Sequence[_RealT] | onp.CanArrayND[_RealT], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[_RealT]: ...
@overload
def diagsvd(s: Sequence[bool], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_]: ...
@overload
def diagsvd(s: Sequence[int], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_ | np.int_]: ...
@overload
def diagsvd(s: Sequence[float], M: op.CanIndex, N: op.CanIndex) -> onp.Array2D[np.bool_ | np.int_ | np.float64]: ...

#
@overload
def orth(A: onp.CanArray2D[_InexactT], rcond: onp.ToFloat | None = None) -> onp.Array2D[_InexactT]: ...
@overload
def orth(A: onp.ToFloat2D, rcond: onp.ToFloat | None = None) -> _Float2D: ...
@overload
def orth(A: onp.ToComplex2D, rcond: onp.ToFloat | None = None) -> _Inexact2D: ...

#
@overload
def null_space(
    A: onp.CanArray2D[_InexactT],
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.Array2D[_InexactT]: ...
@overload
def null_space(
    A: onp.ToFloat2D,
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Float2D: ...
@overload
def null_space(
    A: onp.ToComplex2D,
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Inexact2D: ...

#
def subspace_angles(A: onp.ToComplex2D, B: onp.ToComplex2D) -> _Float1D: ...
