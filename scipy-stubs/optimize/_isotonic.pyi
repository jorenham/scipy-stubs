from typing import type_check_only

import numpy as np
import numpy.typing as npt
from ._optimize import OptimizeResult

__all__ = ["isotonic_regression"]

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    blocks: npt.NDArray[np.intp]

def isotonic_regression(y: npt.ArrayLike, *, weights: npt.ArrayLike | None = None, increasing: bool = True) -> OptimizeResult: ...
