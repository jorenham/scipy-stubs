from collections.abc import Callable
from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
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

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_nd: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.inexact[npt.NBitBase]]]

def fractional_matrix_power(A: npt.ArrayLike, t: float) -> _Array_fc_2d: ...
@overload
def logm(A: npt.ArrayLike, disp: Literal[True] = True) -> _Array_fc_2d: ...
@overload
def logm(A: npt.ArrayLike, disp: Literal[False]) -> tuple[_Array_fc_2d, float | np.float64]: ...
def expm(A: npt.ArrayLike) -> _Array_fc_nd: ...
def cosm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def sinm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def tanm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def coshm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def sinhm(A: npt.ArrayLike) -> _Array_fc_2d: ...
def tanhm(A: npt.ArrayLike) -> _Array_fc_2d: ...
@overload
def funm(
    A: npt.ArrayLike,
    func: Callable[[npt.NDArray[np.inexact[npt.NBitBase]]], npt.NDArray[np.inexact[npt.NBitBase]]],
    disp: Literal[True] = True,
) -> _Array_fc_2d: ...
@overload
def funm(
    A: npt.ArrayLike,
    func: Callable[[npt.NDArray[np.inexact[npt.NBitBase]]], npt.NDArray[np.inexact[npt.NBitBase]]],
    disp: Literal[False],
) -> tuple[_Array_fc_2d, float | np.float64]: ...
@overload
def signm(A: npt.ArrayLike, disp: Literal[True] = True) -> _Array_fc_2d: ...
@overload
def signm(A: npt.ArrayLike, disp: Literal[False]) -> tuple[_Array_fc_2d, float | np.float64]: ...
def khatri_rao(a: npt.ArrayLike, b: npt.ArrayLike) -> _Array_fc_2d: ...
