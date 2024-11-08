from typing import Any, Literal, type_check_only

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyInt, AnyReal
from scipy.optimize import OptimizeResult

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: npt.NDArray[np.float64]
    fun: float | np.float64
    cost: float | np.float64
    initial_cost: float | np.float64
    optimality: float | np.float64
    active_mask: npt.NDArray[np.float64]
    nit: int
    status: int

# undocumented
def compute_kkt_optimality(
    g: npt.NDArray[np.float64],
    on_bound: npt.NDArray[np.float64],
) -> np.float64: ...

# undocumented
def bvls(
    A: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    x_lsq: npt.NDArray[np.floating[Any]],
    lb: npt.NDArray[np.floating[Any]],
    ub: npt.NDArray[np.floating[Any]],
    tol: AnyReal,
    max_iter: AnyInt | None,
    verbose: Literal[0, 1, 2],
    rcond: AnyReal | None = None,
) -> _OptimizeResult: ...
