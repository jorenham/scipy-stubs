from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from ._expm_frechet import expm_cond, expm_frechet
from ._matfuncs_sqrtm import sqrtm

__all__ = [
    "coshm",
    "cosm",
    "expm",
    "expm_cond",
    "expm_frechet",
    "fractional_matrix_power",
    "funm",
    "khatri_rao",
    "logm",
    "signm",
    "sinhm",
    "sinm",
    "sqrtm",
    "tanhm",
    "tanm",
]

_Array_fc_2d: TypeAlias = onp.Array2D[np.inexact[Any]]
_Array_fc_nd: TypeAlias = onp.ArrayND[np.inexact[Any]]

def fractional_matrix_power(A: npt.ArrayLike, t: float) -> _Array_fc_2d: ...
@overload
def logm(A: npt.ArrayLike, disp: Literal[True] = True) -> _Array_fc_2d: ...
@overload
def logm(A: npt.ArrayLike, disp: Literal[False]) -> tuple[_Array_fc_2d, float | np.float64]: ...
def expm(A: npt.ArrayLike) -> onp.ArrayND[np.inexact[Any]]: ...
def cosm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def sinm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def tanm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def coshm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def sinhm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def tanhm(A: npt.ArrayLike) -> _Array_fc_2d: ...
@overload
def funm(
    A: npt.ArrayLike,
    func: Callable[[_Array_fc_nd], _Array_fc_nd],
    disp: Literal[True, 1] = True,
) -> _Array_fc_2d: ...
@overload
def funm(
    A: npt.ArrayLike,
    func: Callable[[_Array_fc_nd], _Array_fc_nd],
    disp: Literal[False, 0],
) -> tuple[_Array_fc_2d, float | np.float64]: ...
@overload
def signm(A: npt.ArrayLike, disp: Literal[True] = True) -> _Array_fc_2d: ...
@overload
def signm(A: npt.ArrayLike, disp: Literal[False]) -> tuple[_Array_fc_2d, float | np.float64]: ...
def khatri_rao(a: npt.ArrayLike, b: npt.ArrayLike) -> _Array_fc_2d: ...
