from typing import type_check_only

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult

__all__ = ["dual_annealing"]

@type_check_only
class _OptimizeResult(OptimizeResult):
    message: str
    success: bool
    status: int
    fun: float
    x: npt.NDArray[np.float64]  # 1d
    nit: int
    nfev: int

class VisitingDistribution:
    TAIL_LIMIT: float
    MIN_VISIT_BOUND: float
    rand_gen: Untyped
    lower: Untyped
    upper: Untyped
    bound_range: Untyped
    def __init__(self, lb: Untyped, ub: Untyped, visiting_param: Untyped, rand_gen: Untyped) -> None: ...
    def visiting(self, x: Untyped, step: Untyped, temperature: Untyped) -> Untyped: ...
    def visit_fn(self, temperature: Untyped, dim: Untyped) -> Untyped: ...

class EnergyState:
    MAX_REINIT_COUNT: int
    ebest: Untyped
    current_energy: Untyped
    current_location: Untyped
    xbest: Untyped
    lower: Untyped
    upper: Untyped
    callback: Untyped
    def __init__(self, lower: Untyped, upper: Untyped, callback: Untyped | None = None) -> None: ...
    def reset(self, func_wrapper: Untyped, rand_gen: Untyped, x0: Untyped | None = None) -> None: ...
    def update_best(self, e: Untyped, x: Untyped, context: Untyped) -> Untyped: ...
    def update_current(self, e: Untyped, x: Untyped) -> None: ...

class StrategyChain:
    emin: Untyped
    xmin: Untyped
    energy_state: Untyped
    acceptance_param: Untyped
    visit_dist: Untyped
    func_wrapper: Untyped
    minimizer_wrapper: Untyped
    not_improved_idx: int
    not_improved_max_idx: int
    temperature_step: int
    K: Untyped
    def __init__(
        self,
        acceptance_param: Untyped,
        visit_dist: Untyped,
        func_wrapper: Untyped,
        minimizer_wrapper: Untyped,
        rand_gen: Untyped,
        energy_state: Untyped,
    ) -> None: ...
    def accept_reject(self, j: Untyped, e: Untyped, x_visit: Untyped) -> None: ...
    energy_state_improved: bool
    def run(self, step: Untyped, temperature: Untyped) -> Untyped: ...
    def local_search(self) -> Untyped: ...

class ObjectiveFunWrapper:
    func: Untyped
    args: Untyped
    nfev: int
    ngev: int
    nhev: int
    maxfun: Untyped
    def __init__(self, func: Untyped, maxfun: float = 10000000.0, *args: Untyped) -> None: ...
    def fun(self, x: Untyped) -> Untyped: ...

class LocalSearchWrapper:
    LS_MAXITER_RATIO: int
    LS_MAXITER_MIN: int
    LS_MAXITER_MAX: int
    func_wrapper: Untyped
    kwargs: Untyped
    jac: Untyped
    hess: Untyped
    hessp: Untyped
    minimizer: Untyped
    lower: Untyped
    upper: Untyped
    def __init__(self, search_bounds: Untyped, func_wrapper: Untyped, *args: Untyped, **kwargs: Untyped) -> None: ...
    def local_search(self, x: Untyped, e: Untyped) -> Untyped: ...

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
