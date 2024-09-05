from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype as op
import scipy._typing as spt

__all__ = [
    "cdf2rdf",
    "eig",
    "eig_banded",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvals_banded",
    "eigvalsh",
    "eigvalsh_tridiagonal",
    "hessenberg",
]

_Array_f: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating[npt.NBitBase]]]
_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

_EigSelect: TypeAlias = Literal["a", "v", "i"]
_EigSelectRange: TypeAlias = tuple[spt.AnyReal, spt.AnyReal]

_EigHType: TypeAlias = Literal[1, 2, 3]
_EigHSubsetByIndex: TypeAlias = op.CanIter[op.CanNext[op.typing.AnyInt]]
_EigHSubsetByValue: TypeAlias = op.CanIter[op.CanNext[spt.AnyReal]]

_LapackDriverE: TypeAlias = Literal["ev", "evd", "evr", "evx"]
_LapackDriverG: TypeAlias = Literal["gv", "gvd", "gvx"]
_LapackDriverST: TypeAlias = Literal["stemr", "stebz", "sterf", "stev"]
_LapackDriverAuto: TypeAlias = Literal["auto"]

# TODO: narrow the `npt.ArrayLike` to specific n-dimensional array-likes.
# TODO: add overloads for shape and dtype

@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    left: Literal[False] = False,
    right: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> _Array_fc_1d | _Array_fc_2d: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    *,
    left: Literal[True],
    right: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None,
    left: Literal[True],
    right: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    left: Literal[False] = False,
    *,
    right: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None,
    left: Literal[False],
    right: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    *,
    left: Literal[True],
    right: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eig(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None,
    left: Literal[True],
    right: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_Array_fc_1d | _Array_fc_2d, _Array_fc_2d, _Array_fc_2d]: ...
@overload
def eigh(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _LapackDriverE | _LapackDriverG | None = None,
) -> tuple[_Array_fc_1d, _Array_fc_2d]: ...
@overload
def eigh(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _LapackDriverE | _EigHSubsetByValue | None = None,
) -> _Array_fc_1d: ...
@overload
def eig_banded(
    a_band: npt.ArrayLike,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    max_ev: int = 0,
    check_finite: bool = True,
) -> tuple[_Array_fc_1d, _Array_fc_2d]: ...
@overload
def eig_banded(
    a_band: npt.ArrayLike,
    lower: bool = True,
    *,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    max_ev: int = 0,
    check_finite: bool = True,
) -> _Array_fc_1d: ...
@overload
def eig_banded(
    a_band: npt.ArrayLike,
    lower: bool,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    max_ev: int = 0,
    check_finite: bool = True,
) -> _Array_fc_1d: ...
def eigvals(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    overwrite_a: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> _Array_fc_1d | _Array_fc_2d: ...
def eigvalsh(
    a: npt.ArrayLike,
    b: npt.ArrayLike | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _LapackDriverE | _EigHSubsetByValue | None = None,
) -> _Array_fc_1d: ...
def eigvals_banded(
    a_band: npt.ArrayLike,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    check_finite: bool = True,
) -> _Array_fc_1d: ...
def eigvalsh_tridiagonal(
    d: npt.ArrayLike,
    e: npt.ArrayLike,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    check_finite: bool = True,
    tol: spt.AnyReal = 0.0,
    lapack_driver: _LapackDriverST | _LapackDriverAuto = "auto",
) -> _Array_fc_1d: ...
@overload
def eigh_tridiagonal(
    d: npt.ArrayLike,
    e: npt.ArrayLike,
    eigvals_only: Literal[False] = False,
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    check_finite: bool = True,
    tol: spt.AnyReal = 0.0,
    lapack_driver: _LapackDriverST | _LapackDriverAuto = "auto",
) -> tuple[_Array_fc_1d, _Array_fc_2d]: ...
@overload
def eigh_tridiagonal(
    d: npt.ArrayLike,
    e: npt.ArrayLike,
    eigvals_only: Literal[True],
    select: _EigSelect = "a",
    select_range: _EigSelectRange | None = None,
    check_finite: bool = True,
    tol: spt.AnyReal = 0.0,
    lapack_driver: _LapackDriverST | _LapackDriverAuto = "auto",
) -> _Array_fc_1d: ...
@overload
def hessenberg(
    a: npt.ArrayLike,
    calc_q: Literal[False] = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def hessenberg(
    a: npt.ArrayLike,
    calc_q: Literal[True],
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
def cdf2rdf(w: npt.ArrayLike, v: npt.ArrayLike) -> tuple[_Array_f, _Array_f]: ...
