from typing import Any, Literal, type_check_only

import numpy as np
import optype.numpy as onp
from scipy.optimize import OptimizeResult

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: onp.ArrayND[np.float64]
    fun: float | np.float64
    cost: float | np.float64
    initial_cost: float | np.float64
    optimality: float | np.float64
    active_mask: onp.ArrayND[np.float64]
    nit: int
    status: int

# undocumented
def compute_kkt_optimality(
    g: onp.ArrayND[np.float64],
    on_bound: onp.ArrayND[np.float64],
) -> np.float64: ...

# undocumented
def bvls(
    A: onp.ArrayND[np.floating[Any]],
    b: onp.ArrayND[np.floating[Any]],
    x_lsq: onp.ArrayND[np.floating[Any]],
    lb: onp.ArrayND[np.floating[Any]],
    ub: onp.ArrayND[np.floating[Any]],
    tol: onp.ToFloat,
    max_iter: onp.ToInt | None,
    verbose: Literal[0, 1, 2],
    rcond: onp.ToFloat | None = None,
) -> _OptimizeResult: ...
