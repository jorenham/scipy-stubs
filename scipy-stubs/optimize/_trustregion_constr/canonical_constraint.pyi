from scipy._typing import Untyped

class CanonicalConstraint:
    n_eq: Untyped
    n_ineq: Untyped
    fun: Untyped
    jac: Untyped
    hess: Untyped
    keep_feasible: Untyped
    def __init__(self, n_eq, n_ineq, fun, jac, hess, keep_feasible) -> None: ...
    @classmethod
    def from_PreparedConstraint(cls, constraint) -> Untyped: ...
    @classmethod
    def empty(cls, n) -> Untyped: ...
    @classmethod
    def concatenate(cls, canonical_constraints, sparse_jacobian) -> Untyped: ...

def initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian) -> Untyped: ...
