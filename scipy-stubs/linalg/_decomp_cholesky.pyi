from typing import Any

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["cho_factor", "cho_solve", "cho_solve_banded", "cholesky", "cholesky_banded"]

def cholesky(
    a: npt.ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.inexact[Any]]: ...
def cho_factor(
    a: npt.ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.inexact[Any]], bool]: ...
def cho_solve(
    c_and_lower: tuple[npt.ArrayLike, bool],
    b: npt.ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.inexact[Any]]: ...
def cholesky_banded(
    ab: npt.ArrayLike,
    overwrite_ab: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.inexact[Any]]: ...
def cho_solve_banded(
    cb_and_lower: tuple[npt.ArrayLike, bool],
    b: npt.ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.inexact[Any]]: ...
