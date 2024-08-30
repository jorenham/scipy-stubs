from scipy._typing import Untyped
from scipy.sparse.linalg import LinearOperator as LinearOperator
from .equality_constrained_sqp import equality_constrained_sqp as equality_constrained_sqp

class BarrierSubproblem:
    n_vars: Untyped
    x0: Untyped
    s0: Untyped
    fun: Untyped
    grad: Untyped
    lagr_hess: Untyped
    constr: Untyped
    jac: Untyped
    barrier_parameter: Untyped
    tolerance: Untyped
    n_eq: Untyped
    n_ineq: Untyped
    enforce_feasibility: Untyped
    global_stop_criteria: Untyped
    xtol: Untyped
    fun0: Untyped
    grad0: Untyped
    constr0: Untyped
    jac0: Untyped
    terminate: bool
    def __init__(
        self,
        x0,
        s0,
        fun,
        grad,
        lagr_hess,
        n_vars,
        n_ineq,
        n_eq,
        constr,
        jac,
        barrier_parameter,
        tolerance,
        enforce_feasibility,
        global_stop_criteria,
        xtol,
        fun0,
        grad0,
        constr_ineq0,
        jac_ineq0,
        constr_eq0,
        jac_eq0,
    ) -> None: ...
    def update(self, barrier_parameter, tolerance): ...
    def get_slack(self, z) -> Untyped: ...
    def get_variables(self, z) -> Untyped: ...
    def function_and_constraints(self, z) -> Untyped: ...
    def scaling(self, z) -> Untyped: ...
    def gradient_and_jacobian(self, z) -> Untyped: ...
    def lagrangian_hessian_x(self, z, v) -> Untyped: ...
    def lagrangian_hessian_s(self, z, v) -> Untyped: ...
    def lagrangian_hessian(self, z, v) -> Untyped: ...
    def stop_criteria(
        self, state, z, last_iteration_failed, optimality, constr_violation, trust_radius, penalty, cg_info
    ) -> Untyped: ...

def tr_interior_point(
    fun,
    grad,
    lagr_hess,
    n_vars,
    n_ineq,
    n_eq,
    constr,
    jac,
    x0,
    fun0,
    grad0,
    constr_ineq0,
    jac_ineq0,
    constr_eq0,
    jac_eq0,
    stop_criteria,
    enforce_feasibility,
    xtol,
    state,
    initial_barrier_parameter,
    initial_tolerance,
    initial_penalty,
    initial_trust_radius,
    factorization_method,
) -> Untyped: ...
