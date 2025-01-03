from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

__all__ = [
    "det",
    "inv",
    "lstsq",
    "matmul_toeplitz",
    "matrix_balance",
    "pinv",
    "pinvh",
    "solve",
    "solve_banded",
    "solve_circulant",
    "solve_toeplitz",
    "solve_triangular",
    "solveh_banded",
]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]

_Float: TypeAlias = np.floating[Any]
_Float0D: TypeAlias = onp.Array0D[_Float]
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]
_FloatND: TypeAlias = onp.ArrayND[_Float]

_Complex: TypeAlias = np.inexact[Any]  # float and complex input types are near impossible to distinguish
_Complex0D: TypeAlias = onp.Array0D[_Complex]
_Complex1D: TypeAlias = onp.Array1D[_Complex]
_Complex2D: TypeAlias = onp.Array2D[_Complex]
_ComplexND: TypeAlias = onp.ArrayND[_Complex]

_AssumeA: TypeAlias = Literal[
    "diagonal",
    "tridiagonal",
    "banded",
    "upper triangular",
    "lower triangular",
    "symmetric", "sym",
    "hermitian", "her",
    "positive definite", "pos",
    "general", "gen",
]  # fmt: skip
_TransSystem: TypeAlias = Literal[0, "N", 1, "T", 2, "C"]
_Singular: TypeAlias = Literal["lstsq", "raise"]
_LapackDriver: TypeAlias = Literal["gelsd", "gelsy", "gelss"]

###

lapack_cast_dict: Final[dict[str, str]] = ...

@overload  # (float[:, :], float[:, :]) -> float[:, :]
def solve(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _Float2D: ...
@overload  # (complex[:, :], complex[:, :]) -> complex[:, :]
def solve(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _Complex2D: ...

#
@overload  # (float[:, :], float[:]) -> float[:]
def solve_triangular(
    a: onp.ToFloat2D,
    b: onp.ToFloatStrict1D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # (float[:, :], float[:, :]) -> float[:. :]
def solve_triangular(
    a: onp.ToFloat2D,
    b: onp.ToFloatStrict2D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:, :], float[:, :?]) -> float[:. :?]
def solve_triangular(
    a: onp.ToFloat2D,
    b: onp.ToFloat1D | onp.ToFloat2D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D | _Float2D: ...
@overload  # (complex[:, :], complex[:, :?]) -> complex[:. :?]
def solve_triangular(
    a: onp.ToComplex2D,
    b: onp.ToComplex1D | onp.ToComplex2D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D | _Complex2D: ...

#
@overload  # (float[:, :], float[:]) -> float[:]
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToFloat2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # (float[:, :], float[:, :]) -> float[:, :]
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToFloat2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:, :], float[:, :?]) -> float[:, :?]
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToFloat2D,
    b: onp.ToFloat1D | onp.ToFloat2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D | _Float2D: ...
@overload  # (complex[:, :], complex[:, :?]) -> complex[:, :?]
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToComplex2D,
    b: onp.ToComplex1D | onp.ToComplex2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D | _Complex2D: ...

#
@overload  # (float[:, :], float[:]) -> float[:]
def solveh_banded(
    ab: onp.ToFloat2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # (float[:, :], float[:, :]) -> float[:, :]
def solveh_banded(
    ab: onp.ToFloat2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:, :], float[:, :?]) -> float[:, :?]
def solveh_banded(
    ab: onp.ToFloat2D,
    b: onp.ToFloat1D | onp.ToFloat2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D | _Float2D: ...
@overload  # (complex[:, :], complex[:, :?]) -> complex[:, :?]
def solveh_banded(
    ab: onp.ToComplex2D,
    b: onp.ToComplex1D | onp.ToComplex2D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D | _Complex2D: ...

#
@overload  # (float[:], float[:]) -> float[:]
def solve_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    b: onp.ToFloatStrict1D,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # (float[:], float[:, :]) -> float[:, :]
def solve_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    b: onp.ToFloatStrict2D,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:], float[:, :?]) -> float[:, :?]
def solve_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    b: onp.ToFloat1D | onp.ToFloat2D,
    check_finite: onp.ToBool = True,
) -> _Float1D | _Float2D: ...
@overload  # (complex[:], complex[:, :?]) -> complex[:, :?]
def solve_toeplitz(
    c_or_cr: onp.ToComplex1D | _Tuple2[onp.ToComplex1D],
    b: onp.ToComplex1D | onp.ToComplex2D,
    check_finite: onp.ToBool = True,
) -> _Complex1D | _Complex2D: ...

#
@overload  # (float[:, :], float[:, :]) -> float[:]
def solve_circulant(
    c: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> _Float1D: ...
@overload  # (float[:, :, ...], float[:, :, ...]) -> float[:, ...]
def solve_circulant(
    c: onp.ToFloatND,
    b: onp.ToFloatND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array[onp.AtLeast1D, _Float]: ...
@overload  # (complex[:, :, ...], complex[:, :, ...]) -> complex[:, ...]
def solve_circulant(
    c: onp.ToComplexND,
    b: onp.ToComplexND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array[onp.AtLeast1D, _Complex]: ...

#
@overload  # float[:, :] -> float[:, :]
def inv(a: onp.ToFloat2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float2D: ...
@overload  # complex[:, :] -> complex[:, :]
def inv(a: onp.ToComplex2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Complex2D: ...

#
@overload  # float[:, :] -> float
def det(a: onp.ToFloatStrict2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float: ...
@overload  # float[:, :, :] -> float[:]
def det(a: onp.ToFloatStrict3D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float1D: ...
@overload  # float[:, :, ...] -> float | float[...]
def det(a: onp.ToFloatND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float | _FloatND: ...
@overload  # complex[:, :, ...] -> complex | complex[...]
def det(a: onp.ToComplexND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Complex | _ComplexND: ...
@overload  # (float[:, :], float[:]) -> (float[:], float[], ...)
def lstsq(
    a: onp.ToFloat2D,
    b: onp.ToFloatStrict1D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Float1D, _Float0D, int, _Float1D | None]: ...
@overload  # (float[:, :], float[:, :]) -> (float[:, :], float[:], ...)
def lstsq(
    a: onp.ToFloat2D,
    b: onp.ToFloatStrict2D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Float2D, _Float1D, int, _Float1D | None]: ...
@overload  # (float[:, :], float[:, :?]) -> (float[:, :?], float[:?], ...)
def lstsq(
    a: onp.ToFloat2D,
    b: onp.ToFloat1D | onp.ToFloat2D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Float1D | _Float2D, _Float0D | _Float1D, int, _Float1D | None]: ...
@overload  # (complex[:, :], complex[:, :?]) -> (complex[:, :?], complex[:?], ...)
def lstsq(
    a: onp.ToComplex2D,
    b: onp.ToComplex1D | onp.ToComplex2D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Complex1D | _Complex2D, _Complex0D | _Complex1D, int, _Complex1D | None]: ...

#
@overload
def pinv(  # (float[:, :], return_rank=False) -> float[:, :]
    a: onp.ToFloat2D,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:, :], return_rank=True) -> (float[:, :], int)
def pinv(
    a: onp.ToFloat2D,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinv(
    a: onp.ToComplex2D,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
@overload  # (complex[:, :], return_rank=True) -> (complex[:, :], int)
def pinv(
    a: onp.ToComplex2D,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, int]: ...

#
@overload  # (float[:, :], return_rank=False) -> float[:, :]
def pinvh(
    a: onp.ToFloat2D,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float[:, :], return_rank=True, /) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloat2D,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: onp.ToBool,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, int]: ...
@overload  # (float[:, :], *, return_rank=True) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloat2D,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    *,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinvh(
    a: onp.ToComplex2D,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
@overload  # (complex[:, :], return_rank=True, /) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplex2D,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: onp.ToBool,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, int]: ...
@overload  # (complex[:, :], *, return_rank=True) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplex2D,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    *,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, int]: ...

#
@overload  # (float[:, :], separate=True) -> (float[:, :], float[:, :])
def matrix_balance(
    A: onp.ToFloat2D,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: Falsy = False,
    overwrite_a: onp.ToBool = False,
) -> _Tuple2[_Float2D]: ...
@overload  # (float[:, :], separate=False, /) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloat2D,
    permute: onp.ToBool,
    scale: onp.ToBool,
    separate: Truthy,
    overwrite_a: onp.ToBool = False,
) -> tuple[_Float2D, _Tuple2[_Float1D]]: ...
@overload  # (float[:, :], *, separate=False) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloat2D,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    *,
    separate: Truthy,
    overwrite_a: onp.ToBool = False,
) -> tuple[_Float2D, _Tuple2[_Float1D]]: ...
@overload  # (complex[:, :], separate=True) -> (complex[:, :], complex[:, :])
def matrix_balance(
    A: onp.ToComplex2D,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: Falsy = False,
    overwrite_a: onp.ToBool = False,
) -> _Tuple2[_Complex2D]: ...
@overload  # (complex[:, :], separate=False, /) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplex2D,
    permute: onp.ToBool,
    scale: onp.ToBool,
    separate: Truthy,
    overwrite_a: onp.ToBool = False,
) -> tuple[_Complex2D, _Tuple2[_Complex1D]]: ...
@overload  # (complex[:, :], *, separate=False) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplex2D,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    *,
    separate: Truthy,
    overwrite_a: onp.ToBool = False,
) -> tuple[_Complex2D, _Tuple2[_Complex1D]]: ...

#
@overload  # (float[:], float[:]) -> float[:]
def matmul_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    x: onp.ToFloatStrict1D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Float1D: ...
@overload  # (float[:], float[:, :]) -> float[:, :]
def matmul_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    x: onp.ToFloatStrict2D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Float2D: ...
@overload  # (float[:], float[:, :?]) -> float[:, :?]
def matmul_toeplitz(
    c_or_cr: onp.ToFloat1D | _Tuple2[onp.ToFloat1D],
    x: onp.ToFloat1D | onp.ToFloat2D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Float1D | _Float2D: ...
@overload  # (complex[:], complex[:, :?]) -> complex[:, :?]
def matmul_toeplitz(
    c_or_cr: onp.ToComplex1D | _Tuple2[onp.ToComplex1D],
    x: onp.ToComplex1D | onp.ToComplex2D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Complex1D | _Complex2D: ...
