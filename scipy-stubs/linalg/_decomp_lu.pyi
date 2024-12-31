from typing import Any, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

__all__ = ["lu", "lu_factor", "lu_solve"]

_IntPND: TypeAlias = onp.ArrayND[np.intp]

_Float1D: TypeAlias = onp.Array1D[np.floating[Any]]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_FloatND: TypeAlias = onp.ArrayND[np.floating[Any], onp.AtLeast2D]

_Complex1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.inexact[Any]]
_ComplexND: TypeAlias = onp.ArrayND[np.inexact[Any], onp.AtLeast2D]

_Trans: TypeAlias = Literal[0, 1, 2]

@overload  # float[:, :] -> (float[:, :], float[:])
def lu_factor(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float1D]: ...
@overload  # complex[:, :] -> (complex[:, :], complex[:])
def lu_factor(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Complex1D]: ...

#
@overload  # (float[:, :], float[:]) -> float[:, :]
def lu_solve(
    lu_and_piv: tuple[_Float2D, _Float1D],
    b: onp.ToFloat1D,
    trans: _Trans = 0,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (complex[:, :], complex[:]) -> complex[:, :]
def lu_solve(
    lu_and_piv: tuple[_Complex2D, _Complex1D],
    b: onp.ToComplex1D,
    trans: _Trans = 0,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...

#
@overload  # (float[:, :], permute_l=False, p_indices=False) -> (float[...], float[...], float[...])
def lu(
    a: onp.ToFloat2D,
    permute_l: Falsy = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    p_indices: Falsy = False,
) -> tuple[_FloatND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=False, p_indices=True, /) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloat2D,
    permute_l: Falsy,
    overwrite_a: onp.ToBool,
    check_finite: onp.ToBool,
    p_indices: Truthy,
) -> tuple[_IntPND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=False, *, p_indices=True) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloat2D,
    permute_l: Falsy = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    *,
    p_indices: Truthy,
) -> tuple[_IntPND, _FloatND, _FloatND]: ...
@overload  # (float[:, :], permute_l=True, p_indices=False) -> (intp[...], float[...], float[...])
def lu(
    a: onp.ToFloat2D,
    permute_l: Truthy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    p_indices: onp.ToBool = False,
) -> tuple[_FloatND, _FloatND]: ...
@overload  # (complex[:, :], permute_l=False, p_indices=False) -> (complex[...], complex[...], complex[...])
def lu(
    a: onp.ToComplex2D,
    permute_l: Falsy = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    p_indices: Falsy = False,
) -> tuple[_ComplexND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=False, p_indices=True, /) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToComplex2D,
    permute_l: Falsy,
    overwrite_a: onp.ToBool,
    check_finite: onp.ToBool,
    p_indices: Truthy,
) -> tuple[_IntPND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=False, *, p_indices=True) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToComplex2D,
    permute_l: Falsy = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    *,
    p_indices: Truthy,
) -> tuple[_IntPND, _ComplexND, _ComplexND]: ...
@overload  # (complex[:, :], permute_l=True, p_indices=False) -> (intp[...], complex[...], complex[...])
def lu(
    a: onp.ToComplex2D,
    permute_l: Truthy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    p_indices: onp.ToBool = False,
) -> tuple[_ComplexND, _ComplexND]: ...
