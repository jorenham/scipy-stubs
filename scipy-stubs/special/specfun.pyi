# This file is not meant for public use and will be removed in SciPy v2.0.0.

import numpy as np
import optype.numpy as onpt
from ._basic import clpmn, lpmn, lpn, lqmn

__all__ = ["clpmn", "lpmn", "lpn", "lqmn", "pbdv"]

# originally defined in scipy/special/_specfun.pyx
def pbdv(
    v: float | np.float64,
    x: float | np.float64,
) -> tuple[onpt.Array[tuple[int], np.float64], onpt.Array[tuple[int], np.float64], float, float]: ...
