from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLike, _ArrayLikeNumber_co
from scipy._typing import AnyInt, AnyReal

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

# scalar types
_Scalar_i: TypeAlias = np.integer[Any]
_Scalar_f: TypeAlias = np.floating[Any]
_Scalar_fc: TypeAlias = np.inexact[Any]
_Scalar_f0: TypeAlias = np.float32 | np.float64
_Scalar_fc0: TypeAlias = _Scalar_f0 | np.complex64 | np.complex128

# input types

# NOTE: `None` not excluded because it could have a special meaning
_Falsy: TypeAlias = Literal[False, 0]
# NOTE: Technically most objects are truthy, but in in almost all cases it's either True, and sometimes 1.
_Truthy: TypeAlias = Literal[True, 1]

# NOTE: only "a", "v" and "i" are documented for the `select` params, but internally 0, 1, and 2 are used, respectively.
_SelectA: TypeAlias = Literal["a", "all", 0]
_SelectV: TypeAlias = Literal["v", "value", 1]
_SelectI: TypeAlias = Literal["i", "index", 2]

# NOTE: `_check_select()` requires the `select_range` array-like to be of `int{16,32,64}` when `select: _SelectIndex`
# https://github.com/jorenham/scipy-stubs/issues/154
# NOTE: This `select_range` parameter type must be of shape `(2,)` and in nondescending order
_SelectRange: TypeAlias = Sequence[float | _Scalar_i | _Scalar_f]
_SelectRangeI: TypeAlias = Sequence[int | np.int16 | np.int32 | np.int64]  # no bool, int8 or unsigned ints

_EigHType: TypeAlias = Literal[1, 2, 3]
_EigHSubsetByIndex: TypeAlias = Iterable[op.typing.AnyInt]
_EigHSubsetByValue: TypeAlias = Iterable[AnyReal]

# LAPACK drivers
_DriverEV: TypeAlias = Literal["ev", "evd", "evx", "evr"]
_DriverGV: TypeAlias = Literal["gv", "gvd", "gvx"]
_DriverSTE: TypeAlias = Literal["stemr", "stebz", "sterf", "stev"]
_DriverAuto: TypeAlias = Literal["auto"]

# output types

_N1: TypeAlias = tuple[int]
_N2: TypeAlias = tuple[int, int]
_N2_: TypeAlias = onpt.AtLeast2D

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])

_SCT_f = TypeVar("_SCT_f", bound=_Scalar_f, default=_Scalar_f0)
_SCT2_f = TypeVar("_SCT2_f", bound=_Scalar_f, default=_Scalar_f0)
_SCT_fc = TypeVar("_SCT_fc", bound=_Scalar_fc, default=_Scalar_fc0)

_Array_f: TypeAlias = onpt.Array[_ShapeT, _SCT_f]
_Array_fc: TypeAlias = onpt.Array[_ShapeT, _SCT_fc]

###

@overload  # left: False = ..., right: False = ...
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    left: _Falsy = False,
    right: _Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> _Array_fc[_N1 | _N2]: ...
@overload  # left: True (positional), right: False = ...
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None,
    left: _Truthy,
    right: _Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2]]: ...
@overload  # left: True (keyword), right: False = ...
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    left: _Truthy,
    right: _Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2]]: ...
@overload  # left: False = ..., right: True (positional)
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None,
    left: _Falsy,
    right: _Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2]]: ...
@overload  # left: False = ..., right: True (keyword)
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    left: _Falsy = False,
    *,
    right: _Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2]]: ...
@overload  # left: True (positional), right: True
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None,
    left: _Truthy,
    right: _Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2], _Array_fc[_N2]]: ...
@overload  # left: True (keyword), right: True (keyword)
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    left: _Truthy,
    right: _Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2], _Array_fc[_N2]]: ...
@overload  # left: CanBool = ..., right: CanBool = ... (catch-all)
def eig(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    left: op.CanBool = False,
    right: op.CanBool = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> (
    _Array_fc[_N1 | _N2]
    | tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2]]
    | tuple[_Array_fc[_N1 | _N2], _Array_fc[_N2], _Array_fc[_N2]]
): ...

#
@overload  # eigvals_only: False = ...
def eigh(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: _Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: True
def eigh(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: _Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: CanBool (catch-all)
def eigh(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    lower: op.CanBool,
    eigvals_only: op.CanBool,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]] | _Array_fc[_N1]: ...

#
@overload  # eigvals_only: False = ..., select: _SelectA = ...
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    eigvals_only: _Falsy = False,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: True  (positional), select: _SelectA = ...
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: True  (keyword), select: _SelectA = ... (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    *,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: False = ..., select: _SelectV (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    eigvals_only: _Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: True  (positional), select: _SelectV (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: True  (keyword), select: _SelectV (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    *,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: False = ..., select: _SelectI (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    eigvals_only: _Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: True (positional), select: _SelectI (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: True (keyword), select: _SelectI (keyword)
def eig_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = True,
    *,
    eigvals_only: _Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: AnyInt = 0,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...

#
def eigvals(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> _Array_fc[_N1 | _N2]: ...

#
def eigvalsh(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co | None = None,
    *,
    lower: op.CanBool = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _Array_fc[_N1]: ...

#
@overload  # select: _SelectA = ...
def eigvals_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectV (positional)
def eigvals_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool,
    overwrite_a_band: op.CanBool,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectV (keyword)
def eigvals_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectI (positional)
def eigvals_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool,
    overwrite_a_band: op.CanBool,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectI (keyword)
def eigvals_banded(
    a_band: _ArrayLikeNumber_co,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N1]: ...

#
@overload  # select: _SelectA = ...
def eigvalsh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectV
def eigvalsh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...
@overload  # select: _SelectI
def eigvalsh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...

#
@overload  # eigvals_only: False = ..., select: _SelectA = ...
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Falsy = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: False, select: _SelectV (positional)
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Falsy,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: False = ..., select: _SelectV (keyword)
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Falsy = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: False, select: _SelectI (positional)
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Falsy,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: False = ..., select: _SelectI (keyword)
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Falsy = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Array_fc[_N1], _Array_fc[_N2]]: ...
@overload  # eigvals_only: True, select: _SelectA = ...
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Truthy,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: True, select: _SelectV
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Truthy,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...
@overload  # eigvals_only: True, select: _SelectI
def eigh_tridiagonal(
    d: _ArrayLikeNumber_co,
    e: _ArrayLikeNumber_co,
    eigvals_only: _Truthy,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: AnyReal = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Array_fc[_N1]: ...

#
@overload  # calc_q: False = ...
def hessenberg(
    a: _ArrayLikeNumber_co,
    calc_q: _Falsy = False,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N2]: ...
@overload  # calc_q: True
def hessenberg(
    a: _ArrayLikeNumber_co,
    calc_q: _Truthy,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> tuple[_Array_fc[_N2], _Array_fc[_N2]]: ...
@overload  # calc_q: CanBool (catch-all)
def hessenberg(
    a: _ArrayLikeNumber_co,
    calc_q: op.CanBool,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> _Array_fc[_N2] | tuple[_Array_fc[_N2], _Array_fc[_N2]]: ...

#
@overload
def cdf2rdf(w: _ArrayLike[_SCT_f], v: _ArrayLike[_SCT2_f]) -> tuple[_Array_f[_N2_, _SCT_f], _Array_f[_N2_, _SCT2_f]]: ...
@overload
def cdf2rdf(w: _ArrayLike[_SCT_f], v: _ArrayLikeNumber_co) -> tuple[_Array_f[_N2_, _SCT_f], _Array_f[_N2_, _Scalar_f]]: ...
@overload
def cdf2rdf(w: _ArrayLikeNumber_co, v: _ArrayLike[_SCT2_f]) -> tuple[_Array_f[_N2_, _Scalar_f], _Array_f[_N2_, _SCT2_f]]: ...
@overload
def cdf2rdf(w: _ArrayLikeNumber_co, v: _ArrayLikeNumber_co) -> tuple[_Array_f[_N2_, _Scalar_f], _Array_f[_N2_, _Scalar_f]]: ...
