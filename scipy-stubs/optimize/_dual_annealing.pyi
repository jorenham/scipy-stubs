from typing import type_check_only

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as _OptimizeResult

__all__ = ["dual_annealing"]

@type_check_only
class OptimizeResult(_OptimizeResult):
    x: onp.Array1D[np.float64]
    fun: float
    status: int
    success: bool
    message: str
    nit: int
    nfev: int

def dual_annealing(
    func: Untyped,
    bounds: Untyped,
    args: Untyped = (),
    maxiter: int = 1000,
    minimizer_kwargs: Untyped | None = None,
    initial_temp: float = 5230.0,
    restart_temp_ratio: float = 2e-05,
    visit: float = 2.62,
    accept: float = -5.0,
    maxfun: float = 10000000.0,
    seed: Untyped | None = None,
    no_local_search: bool = False,
    callback: Untyped | None = None,
    x0: Untyped | None = None,
) -> _OptimizeResult: ...
