from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import scipy._typing as spt

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

_Array_fc: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_0d: TypeAlias = np.ndarray[tuple[()], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

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
) -> _Array_fc_2d: ...
def solve_triangular(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    trans: Literal[0, "N", 1, "T", 2, "C"] = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Array_fc_1d | _Array_fc_2d: ...
def solve_banded(
    l_and_u: npt.ArrayLike,
    ab: npt.ArrayLike,
    b: npt.ArrayLike,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Array_fc_1d | _Array_fc_2d: ...
def solveh_banded(
    ab: npt.ArrayLike,
    b: npt.ArrayLike,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> _Array_fc_1d | _Array_fc_2d: ...
def solve_toeplitz(
    c_or_cr: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    b: npt.ArrayLike,
    check_finite: bool = True,
) -> _Array_fc_1d | _Array_fc_2d: ...
def solve_circulant(
    c: npt.ArrayLike,
    b: npt.ArrayLike,
    singular: Literal["lstsq", "raise"] = "raise",
    tol: spt.AnyReal | None = None,
    caxis: spt.AnyInt = -1,
    baxis: spt.AnyInt = 0,
    outaxis: spt.AnyInt = 0,
) -> _Array_fc: ...
def inv(a: npt.ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> _Array_fc_2d: ...
def det(a: npt.ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> np.inexact[npt.NBitBase] | _Array_fc: ...

# TODO: lstsq.default_lapack_driver
def lstsq(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    cond: spt.AnyReal | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gelsd", "gelsy", "gelss"] | None = None,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_0d | _Array_fc_1d, int, _Array_fc | None]: ...
@overload
def pinv(
    a: npt.ArrayLike,
    *,
    atol: spt.AnyReal | None = None,
    rtol: spt.AnyReal | None = None,
    return_rank: Literal[False] = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def pinv(
    a: npt.ArrayLike,
    *,
    atol: spt.AnyReal | None = None,
    rtol: spt.AnyReal | None = None,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, int]: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: spt.AnyReal | None = None,
    rtol: spt.AnyReal | None = None,
    lower: bool = True,
    return_rank: Literal[False] = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: spt.AnyReal | None = None,
    rtol: spt.AnyReal | None = None,
    lower: bool = True,
    *,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, int]: ...
@overload
def pinvh(
    a: npt.ArrayLike,
    atol: spt.AnyReal | None,
    rtol: spt.AnyReal | None,
    lower: bool,
    return_rank: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, int]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool = True,
    scale: bool = True,
    separate: Literal[False] = False,
    overwrite_a: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool = True,
    scale: bool = True,
    *,
    separate: Literal[True],
    overwrite_a: bool = False,
) -> tuple[_Array_fc_2d, tuple[_Array_fc_1d, _Array_fc_1d]]: ...
@overload
def matrix_balance(
    A: npt.ArrayLike,
    permute: bool,
    scale: bool,
    separate: Literal[True],
    overwrite_a: bool = False,
) -> tuple[_Array_fc_2d, tuple[_Array_fc_1d, _Array_fc_1d]]: ...
def matmul_toeplitz(
    c_or_cr: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    x: npt.ArrayLike,
    check_finite: bool = False,
    workers: int | None = None,
) -> _Array_fc_1d | _Array_fc_2d: ...
