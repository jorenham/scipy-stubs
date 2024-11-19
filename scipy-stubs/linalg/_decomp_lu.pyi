from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["lu", "lu_factor", "lu_solve"]

_Array_i: TypeAlias = onp.ArrayND[np.intp]
_Array_fc: TypeAlias = onp.ArrayND[np.inexact[Any]]
_Array_fc_1d: TypeAlias = onp.Array1D[np.inexact[Any]]
_Array_fc_2d: TypeAlias = onp.Array2D[np.inexact[Any]]

def lu_factor(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_1d]: ...

#
def lu_solve(
    lu_and_piv: tuple[_Array_fc_2d, _Array_fc_1d],
    b: npt.ArrayLike,
    trans: Literal[0, 1, 2] = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...

#
@overload
def lu(
    a: onp.ToComplex2D,
    permute_l: Literal[False, 0] = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: Literal[False] = False,
) -> tuple[_Array_fc, _Array_fc, _Array_fc]: ...
@overload
def lu(
    a: onp.ToComplex2D,
    permute_l: Literal[False],
    overwrite_a: bool,
    check_finite: bool,
    p_indices: Literal[True],
) -> tuple[_Array_i, _Array_fc, _Array_fc]: ...
@overload
def lu(
    a: onp.ToComplex2D,
    permute_l: Literal[False, 0] = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
    *,
    p_indices: Literal[True, 1],
) -> tuple[_Array_i, _Array_fc, _Array_fc]: ...
@overload
def lu(
    a: onp.ToComplex2D,
    permute_l: Literal[True],
    overwrite_a: bool = False,
    check_finite: bool = True,
    p_indices: bool = False,
) -> tuple[_Array_fc, _Array_fc]: ...
