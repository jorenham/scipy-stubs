from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from numpy._typing import _ArrayLike
from scipy._typing import Falsy, Truthy

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

_FloatT = TypeVar("_FloatT", bound=_Floating, default=_Float)
_FloatT2 = TypeVar("_FloatT2", bound=_Floating, default=_Float)

# scalar types
_Integer: TypeAlias = np.integer[Any]
_Floating: TypeAlias = np.floating[Any]
_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

# input types
# NOTE: only "a", "v" and "i" are documented for the `select` params, but internally 0, 1, and 2 are used, respectively.
_SelectA: TypeAlias = Literal["a", "all", 0]
_SelectV: TypeAlias = Literal["v", "value", 1]
_SelectI: TypeAlias = Literal["i", "index", 2]

# NOTE: `_check_select()` requires the `select_range` array-like to be of `int{16,32,64}` when `select: _SelectIndex`
# https://github.com/scipy/scipy-stubs/issues/154
# NOTE: This `select_range` parameter type must be of shape `(2,)` and in nondescending order
_SelectRange: TypeAlias = Sequence[float | _Integer | _Floating]
_SelectRangeI: TypeAlias = Sequence[int | np.int16 | np.int32 | np.int64]  # no bool, int8 or unsigned ints

_EigHType: TypeAlias = Literal[1, 2, 3]
_EigHSubsetByIndex: TypeAlias = Iterable[opt.AnyInt]
_EigHSubsetByValue: TypeAlias = Iterable[onp.ToFloat]

# LAPACK drivers
_DriverGV: TypeAlias = Literal["gv", "gvd", "gvx"]
_DriverEV: TypeAlias = Literal["ev", "evd", "evx", "evr"]
_DriverSTE: TypeAlias = Literal["stemr", "stebz", "sterf", "stev"]
_DriverAuto: TypeAlias = Literal["auto"]

# output types
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]
_Float2ND: TypeAlias = onp.Array[onp.AtLeast2D, _FloatT]

_Complex1D: TypeAlias = onp.Array1D[_Complex]
_Complex2D: TypeAlias = onp.Array2D[_Complex]
_Complex1D2D: TypeAlias = _Complex1D | _Complex2D

_Inexact2D: TypeAlias = onp.Array2D[_Float | _Complex]

###

# `eig` has `2 * (6 + 7) + 1 == 27` overloads...
@overload  # float, left: True (positional), right: True = ..., homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D]: ...
@overload  # float, left: True (positional), right: True = ..., homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D]: ...
@overload  # float, left: True (keyword), right: True = ..., homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D]: ...
@overload  # float, left: True (keyword), right: True = ..., homogeneous_eigvals: True
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D]: ...
@overload  # float, left: False, right: False (positional), homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Falsy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D]: ...
@overload  # float, left: False, right: False (positional), homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Falsy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D]: ...
@overload  # float, left: False = ..., right: False (keyword), homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    left: Falsy = False,
    *,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D]: ...
@overload  # float, left: False = ..., right: False (keyword), homogeneous_eigvals: True
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    left: Falsy = False,
    *,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D]: ...
@overload  # float, left: True (positional), right: False, homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D, _Float2D]: ...
@overload  # float, left: True (positional), right: False, homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D, _Float2D]: ...
@overload  # float, left: True (keyword), right: False, homogeneous_eigvals: False = ...
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Float2D, _Float2D]: ...
@overload  # float, left: True (keyword), right: False, homogeneous_eigvals: True
def eig(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Float2D, _Float2D]: ...
@overload  # complex, left: False = ..., right: True = ..., homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    left: Falsy = False,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> _Complex1D: ...
@overload  # complex, left: False = ..., right: True = ..., homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    left: Falsy = False,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> _Complex2D: ...
@overload  # complex, left: True (positional), right: True = ..., homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D]: ...
@overload  # complex, left: True (positional), right: True = ..., homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D]: ...
@overload  # complex, left: True (keyword), right: True = ..., homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D]: ...
@overload  # complex, left: True (keyword), right: True = ..., homogeneous_eigvals: True
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    left: Truthy,
    right: Truthy = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D]: ...
@overload  # complex, left: False, right: False (positional), homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Falsy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D]: ...
@overload  # complex, left: False, right: False (positional), homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Falsy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D]: ...
@overload  # complex, left: False = ..., right: False (keyword), homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    left: Falsy = False,
    *,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D]: ...
@overload  # complex, left: False = ..., right: False (keyword), homogeneous_eigvals: True
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    left: Falsy = False,
    *,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D]: ...
@overload  # complex, left: True (positional), right: False, homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D, _Inexact2D]: ...
@overload  # complex, left: True (positional), right: False, homogeneous_eigvals: True (keyword)
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D, _Inexact2D]: ...
@overload  # complex, left: True (keyword), right: False (keyword), homogeneous_eigvals: False = ...
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> tuple[_Complex1D, _Inexact2D, _Inexact2D]: ...
@overload  # complex, left: True (keyword), right: False (keyword), homogeneous_eigvals: True
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    left: Truthy,
    right: Falsy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Truthy,
) -> tuple[_Complex2D, _Inexact2D, _Inexact2D]: ...
@overload  # catch-all
def eig(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    left: onp.ToBool = False,
    right: onp.ToBool = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: onp.ToBool = False,
) -> (
    _Complex1D
    | _Complex2D
    | tuple[_Complex1D | _Complex2D, _Inexact2D]
    | tuple[_Complex1D | _Complex2D, _Inexact2D, _Inexact2D]
):  # fmt: skip
    ...

#
@overload  # float, eigvals_only: False = ...
def eigh(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_Float1D, _Float2D]: ...
@overload  # float, eigvals_only: True
def eigh(
    a: onp.ToFloat2D,
    b: onp.ToFloat2D | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _Float1D: ...
@overload  # complex, eigvals_only: False = ...
def eigh(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_Float1D, _Inexact2D]: ...
@overload  # complex, eigvals_only: True
def eigh(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    lower: op.CanBool = True,
    eigvals_only: Truthy,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _Float1D: ...
@overload  # complex, eigvals_only: CanBool (catch-all)
def eigh(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
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
) -> _Float1D | tuple[_Float1D, _Inexact2D]: ...

#
@overload  # float, eigvals_only: False = ..., select: _SelectA = ...
def eig_banded(
    a_band: onp.ToFloat2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Float2D]: ...
@overload  # float, eigvals_only: False = ..., select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToFloat2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Float2D]: ...
@overload  # float, eigvals_only: False = ..., select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToFloat2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Float2D]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectA = ...
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Inexact2D]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Inexact2D]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    eigvals_only: Falsy = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> tuple[_Float1D, _Inexact2D]: ...
@overload  # eigvals_only: True  (positional), select: _SelectA = ...
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # eigvals_only: True  (keyword), select: _SelectA = ... (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    *,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # eigvals_only: True  (positional), select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # eigvals_only: True  (keyword), select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    *,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # eigvals_only: True (positional), select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # eigvals_only: True (keyword), select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = True,
    *,
    eigvals_only: Truthy,
    overwrite_a_band: op.CanBool = False,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: op.CanBool = True,
) -> _Float1D: ...

#
@overload  # homogeneous_eigvals: False = ...
def eigvals(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: Falsy = False,
) -> _Complex1D: ...
@overload  # homogeneous_eigvals: True (positional)
def eigvals(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None,
    overwrite_a: op.CanBool,
    check_finite: op.CanBool,
    homogeneous_eigvals: Truthy,
) -> _Complex2D: ...
@overload  # homogeneous_eigvals: True (keyword)
def eigvals(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
    *,
    homogeneous_eigvals: Truthy,
) -> _Complex2D: ...
@overload  # catch-all
def eigvals(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
    homogeneous_eigvals: op.CanBool = False,
) -> _Complex1D2D: ...

#
def eigvalsh(
    a: onp.ToComplex2D,
    b: onp.ToComplex2D | None = None,
    *,
    lower: op.CanBool = True,
    overwrite_a: op.CanBool = False,
    overwrite_b: op.CanBool = False,
    type: _EigHType = 1,
    check_finite: op.CanBool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _Float1D: ...

#
@overload  # select: _SelectA = ...
def eigvals_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # select: _SelectV (positional)
def eigvals_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool,
    overwrite_a_band: op.CanBool,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # select: _SelectV (keyword)
def eigvals_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # select: _SelectI (positional)
def eigvals_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool,
    overwrite_a_band: op.CanBool,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
) -> _Float1D: ...
@overload  # select: _SelectI (keyword)
def eigvals_banded(
    a_band: onp.ToComplex2D,
    lower: op.CanBool = False,
    overwrite_a_band: op.CanBool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
) -> _Float1D: ...

#
@overload  # select: _SelectA = ...
def eigvalsh_tridiagonal(
    d: onp.ToComplex1D,
    e: onp.ToComplex1D,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...
@overload  # select: _SelectV
def eigvalsh_tridiagonal(
    d: onp.ToComplex1D,
    e: onp.ToComplex1D,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...
@overload  # select: _SelectI
def eigvalsh_tridiagonal(
    d: onp.ToComplex1D,
    e: onp.ToComplex1D,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...

#
@overload  # eigvals_only: False = ..., select: _SelectA = ...
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Falsy = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Float1D, _Float2D]: ...
@overload  # eigvals_only: False, select: _SelectV (positional)
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Falsy,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Float1D, _Float2D]: ...
@overload  # eigvals_only: False = ..., select: _SelectV (keyword)
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Falsy = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Float1D, _Float2D]: ...
@overload  # eigvals_only: False, select: _SelectI (positional)
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Falsy,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Float1D, _Float2D]: ...
@overload  # eigvals_only: False = ..., select: _SelectI (keyword)
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Falsy = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_Float1D, _Float2D]: ...
@overload  # eigvals_only: True, select: _SelectA = ...
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Truthy,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...
@overload  # eigvals_only: True, select: _SelectV
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Truthy,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...
@overload  # eigvals_only: True, select: _SelectI
def eigh_tridiagonal(
    d: onp.ToFloat1D,
    e: onp.ToFloat1D,
    eigvals_only: Truthy,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: op.CanBool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _Float1D: ...

#
@overload  # float, calc_q: False = ...
def hessenberg(
    a: onp.ToFloat2D,
    calc_q: Falsy = False,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> _Float2D: ...
@overload  # float, calc_q: True
def hessenberg(
    a: onp.ToFloat2D,
    calc_q: Truthy,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> tuple[_Float2D, _Float2D]: ...
@overload  # complex, calc_q: False = ...
def hessenberg(
    a: onp.ToComplex2D,
    calc_q: Falsy = False,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> _Inexact2D: ...
@overload  # complex, calc_q: True
def hessenberg(
    a: onp.ToComplex2D,
    calc_q: Truthy,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> tuple[_Inexact2D, _Inexact2D]: ...
@overload  # complex, calc_q: CanBool (catch-all)
def hessenberg(
    a: onp.ToComplex2D,
    calc_q: op.CanBool,
    overwrite_a: op.CanBool = False,
    check_finite: op.CanBool = True,
) -> _Inexact2D | tuple[_Inexact2D, _Inexact2D]: ...

#
@overload
def cdf2rdf(w: _ArrayLike[_FloatT], v: _ArrayLike[_FloatT2]) -> tuple[_Float2ND[_FloatT], _Float2ND[_FloatT2]]: ...
@overload
def cdf2rdf(w: _ArrayLike[_FloatT], v: onp.ToComplexND) -> tuple[_Float2ND[_FloatT], _Float2ND]: ...
@overload
def cdf2rdf(w: onp.ToComplexND, v: _ArrayLike[_FloatT2]) -> tuple[_Float2ND, _Float2ND[_FloatT2]]: ...
@overload
def cdf2rdf(w: onp.ToComplexND, v: onp.ToComplexND) -> tuple[_Float2ND, _Float2ND]: ...
