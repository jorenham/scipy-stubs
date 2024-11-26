from typing import Any

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["ldl"]

def ldl(
    A: npt.ArrayLike,
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[onp.Array2D[np.inexact[Any]], onp.Array2D[np.inexact[Any]], onp.Array1D[np.intp]]: ...
