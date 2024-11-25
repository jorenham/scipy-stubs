from typing import Any, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from numpy._typing import _ArrayLike

__all__ = ["cspline1d", "cspline1d_eval", "gauss_spline", "qspline1d", "qspline1d_eval", "spline_filter"]

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])

@overload
def gauss_spline(x: onp.ToIntND | onp.SequenceND[float], n: onp.ToInt) -> onp.ArrayND[np.float64]: ...
@overload
def gauss_spline(x: onp.SequenceND[complex], n: onp.ToInt) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def gauss_spline(x: _ArrayLike[_SCT_fc], n: onp.ToInt) -> onp.ArrayND[_SCT_fc]: ...

#
def spline_filter(Iin: _ArrayLike[_SCT_fc], lmbda: onp.ToFloat = 5.0) -> onp.ArrayND[_SCT_fc]: ...
def cspline1d(signal: onp.ArrayND[_SCT_fc], lamb: onp.ToFloat = 0.0) -> onp.ArrayND[_SCT_fc]: ...
def qspline1d(signal: onp.ArrayND[_SCT_fc], lamb: onp.ToFloat = 0.0) -> onp.ArrayND[_SCT_fc]: ...
def cspline1d_eval(
    cj: onp.ArrayND[_SCT_fc],
    newx: onp.ToFloatND,
    dx: onp.ToFloat = 1.0,
    x0: onp.ToInt = 0,
) -> onp.ArrayND[_SCT_fc]: ...
def qspline1d_eval(
    cj: onp.ArrayND[_SCT_fc],
    newx: onp.ToFloatND,
    dx: onp.ToFloat = 1.0,
    x0: onp.ToInt = 0,
) -> onp.ArrayND[_SCT_fc]: ...
