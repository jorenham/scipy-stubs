from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["cho_factor", "cho_solve", "cho_solve_banded", "cholesky", "cholesky_banded"]

_Array_fc: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

def cholesky(a: npt.ArrayLike, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True) -> _Array_fc_2d: ...
def cho_factor(
    a: npt.ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc, bool]: ...
def cho_solve(
    c_and_lower: tuple[npt.ArrayLike, bool],
    b: npt.ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Array_fc: ...
def cholesky_banded(
    ab: npt.ArrayLike,
    overwrite_ab: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
def cho_solve_banded(
    cb_and_lower: tuple[npt.ArrayLike, bool],
    b: npt.ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
