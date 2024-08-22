from ._constraints import (
    Bounds as Bounds,
    LinearConstraint as LinearConstraint,
    NonlinearConstraint as NonlinearConstraint,
    PreparedConstraint as PreparedConstraint,
    new_bounds_to_old as new_bounds_to_old,
    new_constraint_to_old as new_constraint_to_old,
    old_bound_to_new as old_bound_to_new,
    old_constraint_to_new as old_constraint_to_new,
)
from ._differentiable_functions import FD_METHODS as FD_METHODS
from ._optimize import MemoizeJac as MemoizeJac, OptimizeResult as OptimizeResult
from scipy._typing import Untyped

MINIMIZE_METHODS: Untyped
MINIMIZE_METHODS_NEW_CB: Untyped
MINIMIZE_SCALAR_METHODS: Untyped

def minimize(
    fun,
    x0,
    args=(),
    method: Untyped | None = None,
    jac: Untyped | None = None,
    hess: Untyped | None = None,
    hessp: Untyped | None = None,
    bounds: Untyped | None = None,
    constraints=(),
    tol: Untyped | None = None,
    callback: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def minimize_scalar(
    fun,
    bracket: Untyped | None = None,
    bounds: Untyped | None = None,
    args=(),
    method: Untyped | None = None,
    tol: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def standardize_bounds(bounds, x0, meth) -> Untyped: ...
def standardize_constraints(constraints, x0, meth) -> Untyped: ...
