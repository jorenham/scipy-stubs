from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

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

_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]
_Inexact0D: TypeAlias = onp.ArrayND[np.inexact[Any], tuple[()]]
_Inexact1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

lapack_cast_dict: dict[str, str]

# TODO: narrow the `npt.ArrayLike` to specific n-dimensional array-likes.
# TODO: add overloads for shape and dtype

def solve(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: Literal["gen", "sym", "her", "pos"] = "gen",
    transposed: bool = False,
) -> _Inexact2D: ...
def solve_triangular(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    trans: Literal[0, "N", 1, "T", 2, "C"] = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Inexact1D | _Inexact2D: ...
def solve_banded(
    l_and_u: npt.ArrayLike,
    ab: npt.ArrayLike,
    b: npt.ArrayLike,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Inexact1D | _Inexact2D: ...
def solveh_banded(
    ab: npt.ArrayLike,
    b: npt.ArrayLike,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> _Inexact1D | _Inexact2D: ...
def solve_toeplitz(
    c_or_cr: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    b: npt.ArrayLike,
    check_finite: bool = True,
) -> _Inexact1D | _Inexact2D: ...
def solve_circulant(
    c: npt.ArrayLike,
    b: npt.ArrayLike,
    singular: Literal["lstsq", "raise"] = "raise",
    tol: onp.ToFloat | None = None,
    caxis: onp.ToInt = -1,
    baxis: onp.ToInt = 0,
    outaxis: onp.ToInt = 0,
) -> _InexactND: ...
def inv(a: npt.ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> _Inexact2D: ...
def det(a: npt.ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> np.inexact[Any] | _InexactND: ...

# TODO: lstsq.default_lapack_driver
def lstsq(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    cond: onp.ToFloat | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gelsd", "gelsy", "gelss"] | None = None,
) -> tuple[_Inexact1D | _Inexact2D, _Inexact0D | _Inexact1D, int, _InexactND | None]: ...
@overload
def pinv(
    a: npt.ArrayLike,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Literal[False] = False,
    check_finite: bool = True,
) -> _Inexact2D: ...
@overload
def pinv(
    a: npt.ArrayLike,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, int]: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    return_rank: Literal[False] = False,
    check_finite: bool = True,
) -> _Inexact2D: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    *,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, int]: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: bool,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, int]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool = True,
    scale: bool = True,
    separate: Literal[False] = False,
    overwrite_a: bool = False,
) -> tuple[_Inexact2D, _Inexact2D]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool = True,
    scale: bool = True,
    *,
    separate: Literal[True],
    overwrite_a: bool = False,
) -> tuple[_Inexact2D, tuple[_Inexact1D, _Inexact1D]]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool,
    scale: bool,
    separate: Literal[True],
    overwrite_a: bool = False,
) -> tuple[_Inexact2D, tuple[_Inexact1D, _Inexact1D]]: ...
def matmul_toeplitz(
    c_or_cr: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    x: npt.ArrayLike,
    check_finite: bool = False,
    workers: int | None = None,
) -> _Inexact1D | _Inexact2D: ...
