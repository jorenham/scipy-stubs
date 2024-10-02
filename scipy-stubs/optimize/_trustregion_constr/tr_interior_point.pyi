from scipy._typing import Untyped

__all__ = ["tr_interior_point"]

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
        x0: Untyped,
        s0: Untyped,
        fun: Untyped,
        grad: Untyped,
        lagr_hess: Untyped,
        n_vars: Untyped,
        n_ineq: Untyped,
        n_eq: Untyped,
        constr: Untyped,
        jac: Untyped,
        barrier_parameter: Untyped,
        tolerance: Untyped,
        enforce_feasibility: Untyped,
        global_stop_criteria: Untyped,
        xtol: Untyped,
        fun0: Untyped,
        grad0: Untyped,
        constr_ineq0: Untyped,
        jac_ineq0: Untyped,
        constr_eq0: Untyped,
        jac_eq0: Untyped,
    ) -> None: ...
    def update(self, barrier_parameter: Untyped, tolerance: Untyped) -> None: ...
    def get_slack(self, z: Untyped) -> Untyped: ...
    def get_variables(self, z: Untyped) -> Untyped: ...
    def function_and_constraints(self, z: Untyped) -> Untyped: ...
    def scaling(self, z: Untyped) -> Untyped: ...
    def gradient_and_jacobian(self, z: Untyped) -> Untyped: ...
    def lagrangian_hessian_x(self, z: Untyped, v: Untyped) -> Untyped: ...
    def lagrangian_hessian_s(self, z: Untyped, v: Untyped) -> Untyped: ...
    def lagrangian_hessian(self, z: Untyped, v: Untyped) -> Untyped: ...
    def stop_criteria(
        self,
        state: Untyped,
        z: Untyped,
        last_iteration_failed: Untyped,
        optimality: Untyped,
        constr_violation: Untyped,
        trust_radius: Untyped,
        penalty: Untyped,
        cg_info: Untyped,
    ) -> Untyped: ...

def tr_interior_point(
    fun: Untyped,
    grad: Untyped,
    lagr_hess: Untyped,
    n_vars: Untyped,
    n_ineq: Untyped,
    n_eq: Untyped,
    constr: Untyped,
    jac: Untyped,
    x0: Untyped,
    fun0: Untyped,
    grad0: Untyped,
    constr_ineq0: Untyped,
    jac_ineq0: Untyped,
    constr_eq0: Untyped,
    jac_eq0: Untyped,
    stop_criteria: Untyped,
    enforce_feasibility: Untyped,
    xtol: Untyped,
    state: Untyped,
    initial_barrier_parameter: Untyped,
    initial_tolerance: Untyped,
    initial_penalty: Untyped,
    initial_trust_radius: Untyped,
    factorization_method: Untyped,
) -> Untyped: ...
