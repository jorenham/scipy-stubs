from scipy._typing import Untyped
from scipy.optimize._constraints import (
    Bounds as Bounds,
    LinearConstraint as LinearConstraint,
    NonlinearConstraint as NonlinearConstraint,
    PreparedConstraint as PreparedConstraint,
    strict_bounds as strict_bounds,
)
from scipy.optimize._differentiable_functions import ScalarFunction as ScalarFunction, VectorFunction as VectorFunction
from scipy.optimize._hessian_update_strategy import BFGS as BFGS
from scipy.optimize._optimize import OptimizeResult as OptimizeResult
from scipy.sparse.linalg import LinearOperator as LinearOperator
from .canonical_constraint import (
    CanonicalConstraint as CanonicalConstraint,
    initial_constraints_as_canonical as initial_constraints_as_canonical,
)
from .equality_constrained_sqp import equality_constrained_sqp as equality_constrained_sqp
from .report import BasicReport as BasicReport, IPReport as IPReport, SQPReport as SQPReport
from .tr_interior_point import tr_interior_point as tr_interior_point

TERMINATION_MESSAGES: Untyped

class HessianLinearOperator:
    hessp: Untyped
    n: Untyped
    def __init__(self, hessp, n) -> None: ...
    def __call__(self, x, *args) -> Untyped: ...

class LagrangianHessian:
    n: Untyped
    objective_hess: Untyped
    constraints_hess: Untyped
    def __init__(self, n, objective_hess, constraints_hess) -> None: ...
    def __call__(self, x, v_eq, v_ineq: Untyped | None = None) -> Untyped: ...

def update_state_sqp(
    state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info
) -> Untyped: ...
def update_state_ip(
    state,
    x,
    last_iteration_failed,
    objective,
    prepared_constraints,
    start_time,
    tr_radius,
    constr_penalty,
    cg_info,
    barrier_parameter,
    barrier_tolerance,
) -> Untyped: ...
