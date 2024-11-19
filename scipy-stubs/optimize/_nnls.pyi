import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["nnls"]

def nnls(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    maxiter: onp.ToInt | None = None,
    *,
    atol: onp.ToFloat | None = None,
) -> tuple[npt.NDArray[np.float64], np.float64]: ...
