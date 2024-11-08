import numpy as np
import numpy.typing as npt
from scipy._typing import AnyInt, AnyReal

__all__ = ["nnls"]

def nnls(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    maxiter: AnyInt | None = None,
    *,
    atol: AnyReal | None = None,
) -> tuple[npt.NDArray[np.float64], np.float64]: ...
