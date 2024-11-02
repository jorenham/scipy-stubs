from typing import Any, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal

__all__ = ["geometric_slerp"]

@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: float | np.floating[Any],
    tol: AnyReal = 1e-07,
) -> onpt.Array[tuple[int], np.float64]: ...
@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: npt.NDArray[np.floating[Any]],
    tol: AnyReal = 1e-07,
) -> onpt.Array[tuple[int, int], np.float64]: ...
@overload
def geometric_slerp(
    start: _ArrayLikeFloat_co,
    end: _ArrayLikeFloat_co,
    t: _ArrayLikeFloat_co,
    tol: AnyReal = 1e-07,
) -> onpt.Array[tuple[int], np.float64] | onpt.Array[tuple[int, int], np.float64]: ...
