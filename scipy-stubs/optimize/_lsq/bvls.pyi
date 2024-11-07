from typing import Any, Literal, type_check_only

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyInt, AnyReal, Untyped
from scipy.optimize import OptimizeResult

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: npt.NDArray[np.float64]
    fun: float | np.float64
    cost: Untyped
    optimality: Untyped
    active_mask: Untyped
    nit: int
    status: int
    initial_cost: Untyped

# undocumented
def compute_kkt_optimality(
    g: npt.NDArray[np.floating[Any]] | AnyReal,
    on_bound: AnyReal,
) -> np.floating[Any] | float: ...

# undocumented
def bvls(
    A: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    x_lsq: npt.NDArray[np.floating[Any]],
    lb: npt.NDArray[np.floating[Any]],
    ub: npt.NDArray[np.floating[Any]],
    tol: AnyReal,
    max_iter: AnyInt,
    verbose: Literal[0, 1, 2],
    rcond: AnyReal | None = None,
) -> _OptimizeResult: ...
