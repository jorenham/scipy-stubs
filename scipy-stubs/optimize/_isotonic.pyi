from typing import Final, final

import numpy as np
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co
from ._optimize import OptimizeResult as _OptimizeResult

__all__ = ["isotonic_regression"]

@final
class OptimizeResult(_OptimizeResult):
    x: Final[onp.Array1D[np.float64]]
    weights: Final[onp.Array1D[np.float64]]
    blocks: Final[onp.Array1D[np.intp]]

def isotonic_regression(
    y: _ArrayLikeFloat_co,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    increasing: op.CanBool = True,
) -> OptimizeResult: ...
