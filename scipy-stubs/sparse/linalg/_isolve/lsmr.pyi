from typing import Literal, TypeAlias

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Floating, Integer
from scipy.sparse.linalg import LinearOperator

__all__ = ["lsmr"]

_Float64: TypeAlias = float | np.float64
_Real: TypeAlias = np.bool_ | Integer | Floating
_ToRealMatrix: TypeAlias = onp.CanArrayND[_Real] | _spbase[_Real] | LinearOperator[_Real]

_IStop: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]

###

def lsmr(
    A: _ToRealMatrix,
    b: onp.ToFloat1D,
    damp: float | Floating = 0.0,
    atol: float | Floating = 1e-6,
    btol: float | Floating = 1e-6,
    conlim: onp.ToFloat = 1e8,
    maxiter: int | None = None,
    show: onp.ToBool = False,
    x0: onp.ToFloat1D | None = None,
) -> tuple[
    onp.Array1D[np.float64],  # x
    _IStop,  # istop
    int,  # itn
    _Float64,  # normr
    _Float64,  # normar
    _Float64,  # norma
    _Float64,  # conda
    _Float64,  # normx
]: ...
