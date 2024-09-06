import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeFloat_co

__all__ = ["log_softmax", "logsumexp", "softmax"]

def logsumexp(
    a: _ArrayLikeComplex_co,
    axis: int | tuple[int, ...] | None = None,
    b: _ArrayLikeFloat_co | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
) -> np.float64 | np.complex128 | npt.NDArray[np.float64 | np.complex128]: ...
def softmax(
    x: _ArrayLikeComplex_co,
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | np.complex128 | npt.NDArray[np.float64 | np.complex128]: ...
def log_softmax(
    x: _ArrayLikeComplex_co,
    axis: int | tuple[int, ...] | None = None,
) -> np.float64 | np.complex128 | npt.NDArray[np.float64 | np.complex128]: ...
