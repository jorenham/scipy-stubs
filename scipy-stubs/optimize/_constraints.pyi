from scipy._typing import Untyped

__all__ = (
    "Bounds",
    "LinearConstraint",
    "NonlinearConstraint",
    "PreparedConstraint",
    "new_bounds_to_old",
    "new_constraint_to_old",
    "old_bound_to_new",
    "old_constraint_to_new",
    "strict_bounds",
)

class NonlinearConstraint:
    fun: Untyped
    lb: Untyped
    ub: Untyped
    finite_diff_rel_step: Untyped
    finite_diff_jac_sparsity: Untyped
    jac: Untyped
    hess: Untyped
    keep_feasible: Untyped
    def __init__(
        self,
        fun,
        lb,
        ub,
        jac: str = "2-point",
        hess: Untyped | None = None,
        keep_feasible: bool = False,
        finite_diff_rel_step: Untyped | None = None,
        finite_diff_jac_sparsity: Untyped | None = None,
    ): ...

class LinearConstraint:
    A: Untyped
    lb: Untyped
    ub: Untyped
    keep_feasible: Untyped
    def __init__(self, A, lb=..., ub=..., keep_feasible: bool = False): ...
    def residual(self, x) -> Untyped: ...

class Bounds:
    lb: Untyped
    ub: Untyped
    keep_feasible: Untyped
    def __init__(self, lb=..., ub=..., keep_feasible: bool = False): ...
    def residual(self, x) -> Untyped: ...

class PreparedConstraint:
    fun: Untyped
    bounds: Untyped
    keep_feasible: Untyped
    def __init__(self, constraint, x0, sparse_jacobian: Untyped | None = None, finite_diff_bounds=...): ...
    def violation(self, x) -> Untyped: ...

def new_bounds_to_old(lb, ub, n) -> Untyped: ...
def old_bound_to_new(bounds) -> Untyped: ...
def strict_bounds(lb, ub, keep_feasible, n_vars) -> Untyped: ...
def new_constraint_to_old(con, x0) -> Untyped: ...
def old_constraint_to_new(ic, con) -> Untyped: ...
