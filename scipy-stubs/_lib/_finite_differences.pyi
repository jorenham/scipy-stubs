from collections.abc import Callable
from typing import Any, Concatenate, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp

_T = TypeVar("_T", bound=np.floating[Any] | onp.ArrayND[np.floating[Any]])

###

def _central_diff_weights(Np: int, ndiv: int = 1) -> onp.Array1D[np.float64]: ...

#
@overload
def _derivative(
    func: Callable[Concatenate[float, ...], onp.ToFloat],
    x0: float,
    dx: onp.ToFloat = 1.0,
    n: onp.ToInt = 1,
    args: tuple[object, ...] = (),
    order: onp.ToInt = 3,
) -> np.float64: ...
@overload
def _derivative(
    func: Callable[Concatenate[_T, ...], onp.ToFloat],
    x0: _T,
    dx: onp.ToFloat = 1.0,
    n: onp.ToInt = 1,
    args: tuple[object, ...] = (),
    order: onp.ToInt = 3,
) -> _T: ...
