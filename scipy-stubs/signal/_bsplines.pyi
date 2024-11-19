from typing import Any, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLike, _ArrayLikeComplex_co, _ArrayLikeFloat_co, _ArrayLikeInt_co

__all__ = ["cspline1d", "cspline1d_eval", "gauss_spline", "qspline1d", "qspline1d_eval", "spline_filter"]

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])

@overload
def gauss_spline(x: _ArrayLikeInt_co, n: onp.ToInt) -> npt.NDArray[np.float64]: ...
@overload
def gauss_spline(x: _ArrayLikeFloat_co, n: onp.ToInt) -> npt.NDArray[np.floating[Any]]: ...
@overload
def gauss_spline(x: _ArrayLike[_SCT_fc], n: onp.ToInt) -> npt.NDArray[_SCT_fc]: ...
@overload
def gauss_spline(x: _ArrayLikeComplex_co, n: onp.ToInt) -> npt.NDArray[np.inexact[Any]]: ...
def spline_filter(Iin: _ArrayLike[_SCT_fc], lmbda: onp.ToFloat = 5.0) -> npt.NDArray[_SCT_fc]: ...
def cspline1d(signal: npt.NDArray[_SCT_fc], lamb: onp.ToFloat = 0.0) -> npt.NDArray[_SCT_fc]: ...
def qspline1d(signal: npt.NDArray[_SCT_fc], lamb: onp.ToFloat = 0.0) -> npt.NDArray[_SCT_fc]: ...
def cspline1d_eval(
    cj: npt.NDArray[_SCT_fc],
    newx: _ArrayLikeFloat_co,
    dx: onp.ToFloat = 1.0,
    x0: onp.ToInt = 0,
) -> npt.NDArray[_SCT_fc]: ...
def qspline1d_eval(
    cj: npt.NDArray[_SCT_fc],
    newx: _ArrayLikeFloat_co,
    dx: onp.ToFloat = 1.0,
    x0: onp.ToInt = 0,
) -> npt.NDArray[_SCT_fc]: ...
