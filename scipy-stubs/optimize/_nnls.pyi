import numpy as np
import optype.numpy as onp

__all__ = ["nnls"]

def nnls(
    A: onp.ToFloat2D,
    b: onp.ToFloat2D,
    maxiter: onp.ToInt | None = None,
    *,
    atol: onp.ToFloat | None = None,
) -> tuple[onp.ArrayND[np.float64], np.float64]: ...
