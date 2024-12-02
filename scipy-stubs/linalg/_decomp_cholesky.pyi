from typing import Any, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["cho_factor", "cho_solve", "cho_solve_banded", "cholesky", "cholesky_banded"]

_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.inexact[Any]]
_ComplexND: TypeAlias = onp.ArrayND[np.inexact[Any]]

###

@overload
def cholesky(
    a: onp.ToFloat2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload
def cholesky(
    a: onp.ToComplex2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...

#
@overload
def cho_factor(
    a: onp.ToFloat2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_FloatND, bool]: ...
@overload
def cho_factor(
    a: onp.ToComplex2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_ComplexND, bool]: ...

#
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToFloat2D, onp.ToBool],
    b: onp.ToFloat1D,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToComplex2D, onp.ToBool],
    b: onp.ToComplex1D,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...

#
@overload
def cholesky_banded(
    ab: onp.ToFloat2D,
    overwrite_ab: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload
def cholesky_banded(
    ab: onp.ToComplex2D,
    overwrite_ab: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...

#
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToFloat2D, onp.ToBool],
    b: onp.ToComplex1D,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplex2D, onp.ToBool],
    b: onp.ToComplex1D,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
