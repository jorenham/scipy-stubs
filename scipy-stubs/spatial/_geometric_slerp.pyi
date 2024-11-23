from typing import Any, overload

import numpy as np
import optype.numpy as onp

__all__ = ["geometric_slerp"]

@overload
def geometric_slerp(
    start: onp.ToFloat1D,
    end: onp.ToFloat1D,
    t: float | np.floating[Any],
    tol: onp.ToFloat = 1e-07,
) -> onp.Array1D[np.float64]: ...
@overload
def geometric_slerp(
    start: onp.ToFloat1D,
    end: onp.ToFloat1D,
    t: onp.ArrayND[np.floating[Any]],
    tol: onp.ToFloat = 1e-07,
) -> onp.Array2D[np.float64]: ...
@overload
def geometric_slerp(
    start: onp.ToFloat1D,
    end: onp.ToFloat1D,
    t: onp.ToFloat1D,
    tol: onp.ToFloat = 1e-07,
) -> onp.Array1D[np.float64] | onp.Array2D[np.float64]: ...
