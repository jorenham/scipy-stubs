from scipy._typing import Untyped, UntypedCallable

class NonlinearConstraint:
    fun: UntypedCallable
    lb: Untyped
    ub: Untyped
    finite_diff_rel_step: Untyped
    finite_diff_jac_sparsity: Untyped
    jac: Untyped
    hess: Untyped
    keep_feasible: Untyped
    def __init__(
        self,
        /,
        fun: UntypedCallable,
        lb: Untyped,
        ub: Untyped,
        jac: str = "2-point",
        hess: Untyped = ...,
        keep_feasible: bool = False,
        finite_diff_rel_step: Untyped | None = None,
        finite_diff_jac_sparsity: Untyped | None = None,
    ) -> None: ...

class LinearConstraint:
    A: Untyped
    lb: Untyped
    ub: Untyped
    keep_feasible: Untyped
    def __init__(self, /, A: Untyped, lb: Untyped = ..., ub: Untyped = ..., keep_feasible: bool = False) -> None: ...
    def residual(self, /, x: Untyped) -> Untyped: ...

class Bounds:
    lb: Untyped
    ub: Untyped
    keep_feasible: Untyped
    def __init__(self, /, lb: Untyped = ..., ub: Untyped = ..., keep_feasible: bool = False) -> None: ...
    def residual(self, /, x: Untyped) -> Untyped: ...

class PreparedConstraint:
    fun: Untyped
    bounds: Untyped
    keep_feasible: Untyped
    def __init__(
        self,
        /,
        constraint: Untyped,
        x0: Untyped,
        sparse_jacobian: Untyped | None = None,
        finite_diff_bounds: Untyped = ...,
    ) -> None: ...
    def violation(self, x: Untyped) -> Untyped: ...

def new_bounds_to_old(lb: Untyped, ub: Untyped, n: Untyped) -> Untyped: ...
def old_bound_to_new(bounds: Untyped) -> Untyped: ...
def strict_bounds(lb: Untyped, ub: Untyped, keep_feasible: Untyped, n_vars: Untyped) -> Untyped: ...
def new_constraint_to_old(con: Untyped, x0: Untyped) -> Untyped: ...
def old_constraint_to_new(ic: Untyped, con: Untyped) -> Untyped: ...
