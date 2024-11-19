from typing import Any, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co

__all__ = ["geometric_slerp"]

@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: float | np.floating[Any],
    tol: onp.ToFloat = 1e-07,
) -> onp.Array1D[np.float64]: ...
@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: npt.NDArray[np.floating[Any]],
    tol: onp.ToFloat = 1e-07,
) -> onp.Array2D[np.float64]: ...
@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: _ArrayLikeFloat_co,
    tol: onp.ToFloat = 1e-07,
) -> onp.Array1D[np.float64] | onp.Array2D[np.float64]: ...
