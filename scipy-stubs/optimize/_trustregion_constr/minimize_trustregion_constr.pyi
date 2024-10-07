from scipy._typing import Untyped, UntypedCallable
from scipy.optimize._optimize import OptimizeResult

TERMINATION_MESSAGES: Untyped

class HessianLinearOperator:
    hessp: Untyped
    n: Untyped
    def __init__(self, hessp: UntypedCallable, n: int) -> None: ...
    def __call__(self, x: Untyped, *args: object) -> Untyped: ...

class LagrangianHessian:
    n: Untyped
    objective_hess: Untyped
    constraints_hess: Untyped
    def __init__(self, n: int, objective_hess: UntypedCallable, constraints_hess: UntypedCallable) -> None: ...
    def __call__(self, x: Untyped, v_eq: Untyped, v_ineq: Untyped) -> Untyped: ...

def update_state_sqp(
    state: Untyped,
    x: Untyped,
    last_iteration_failed: Untyped,
    objective: Untyped,
    prepared_constraints: Untyped,
    start_time: Untyped,
    tr_radius: Untyped,
    constr_penalty: Untyped,
    cg_info: Untyped,
) -> Untyped: ...
def update_state_ip(
    state: Untyped,
    x: Untyped,
    last_iteration_failed: Untyped,
    objective: Untyped,
    prepared_constraints: Untyped,
    start_time: Untyped,
    tr_radius: Untyped,
    constr_penalty: Untyped,
    cg_info: Untyped,
    barrier_parameter: Untyped,
    barrier_tolerance: Untyped,
) -> Untyped: ...
def _minimize_trustregion_constr(
    fun: UntypedCallable,
    x0: Untyped,
    args: Untyped,
    grad: Untyped,
    hess: Untyped,
    hessp: Untyped,
    bounds: Untyped,
    constraints: Untyped,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    barrier_tol: float = 1e-8,
    sparse_jacobian: bool | None = None,
    callback: UntypedCallable | None = None,
    maxiter: int = 1000,
    verbose: Untyped = 0,
    finite_diff_rel_step: Untyped | None = None,
    initial_constr_penalty: float = 1.0,
    initial_tr_radius: float = 1.0,
    initial_barrier_parameter: float = 0.1,
    initial_barrier_tolerance: float = 0.1,
    factorization_method: str | None = None,
    disp: bool = False,
) -> OptimizeResult: ...
