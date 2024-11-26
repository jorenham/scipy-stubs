from typing import Any

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["orthogonal_procrustes"]

def orthogonal_procrustes(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    check_finite: bool = True,
) -> tuple[onp.Array2D[np.inexact[Any]], float | np.float64]: ...
