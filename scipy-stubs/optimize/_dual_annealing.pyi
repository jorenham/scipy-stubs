from scipy._lib._util import check_random_state as check_random_state
from scipy._typing import Untyped
from scipy.optimize import Bounds as Bounds, OptimizeResult as OptimizeResult, minimize as minimize
from scipy.optimize._constraints import new_bounds_to_old as new_bounds_to_old
from scipy.special import gammaln as gammaln

class VisitingDistribution:
    TAIL_LIMIT: float
    MIN_VISIT_BOUND: float
    rand_gen: Untyped
    lower: Untyped
    upper: Untyped
    bound_range: Untyped
    def __init__(self, lb, ub, visiting_param, rand_gen) -> None: ...
    def visiting(self, x, step, temperature) -> Untyped: ...
    def visit_fn(self, temperature, dim) -> Untyped: ...

class EnergyState:
    MAX_REINIT_COUNT: int
    ebest: Untyped
    current_energy: Untyped
    current_location: Untyped
    xbest: Untyped
    lower: Untyped
    upper: Untyped
    callback: Untyped
    def __init__(self, lower, upper, callback: Untyped | None = None): ...
    def reset(self, func_wrapper, rand_gen, x0: Untyped | None = None): ...
    def update_best(self, e, x, context) -> Untyped: ...
    def update_current(self, e, x): ...

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
    def __init__(self, acceptance_param, visit_dist, func_wrapper, minimizer_wrapper, rand_gen, energy_state) -> None: ...
    def accept_reject(self, j, e, x_visit): ...
    energy_state_improved: bool
    def run(self, step, temperature) -> Untyped: ...
    def local_search(self) -> Untyped: ...

class ObjectiveFunWrapper:
    func: Untyped
    args: Untyped
    nfev: int
    ngev: int
    nhev: int
    maxfun: Untyped
    def __init__(self, func, maxfun: float = 10000000.0, *args): ...
    def fun(self, x) -> Untyped: ...

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
    def __init__(self, search_bounds, func_wrapper, *args, **kwargs) -> None: ...
    def local_search(self, x, e) -> Untyped: ...

def dual_annealing(
    func,
    bounds,
    args=(),
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
) -> Untyped: ...
