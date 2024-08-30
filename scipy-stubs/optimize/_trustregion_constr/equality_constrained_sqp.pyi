from scipy._typing import Untyped
from .projections import projections as projections
from .qp_subproblem import (
    box_intersections as box_intersections,
    modified_dogleg as modified_dogleg,
    projected_cg as projected_cg,
)

def default_scaling(x) -> Untyped: ...
def equality_constrained_sqp(
    fun_and_constr,
    grad_and_jac,
    lagr_hess,
    x0,
    fun0,
    grad0,
    constr0,
    jac0,
    stop_criteria,
    state,
    initial_penalty,
    initial_trust_radius,
    factorization_method,
    trust_lb: Untyped | None = None,
    trust_ub: Untyped | None = None,
    scaling=...,
) -> Untyped: ...
