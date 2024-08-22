from ._trustregion import BaseQuadraticSubproblem as BaseQuadraticSubproblem
from scipy._typing import Untyped
from scipy.linalg import (
    cho_solve as cho_solve,
    get_lapack_funcs as get_lapack_funcs,
    norm as norm,
    solve_triangular as solve_triangular,
)

def estimate_smallest_singular_value(U) -> Untyped: ...
def gershgorin_bounds(H) -> Untyped: ...
def singular_leading_submatrix(A, U, k) -> Untyped: ...

class IterativeSubproblem(BaseQuadraticSubproblem):
    UPDATE_COEFF: float
    EPS: Untyped
    previous_tr_radius: int
    lambda_lb: Untyped
    niter: int
    k_easy: Untyped
    k_hard: Untyped
    dimension: Untyped
    hess_inf: Untyped
    hess_fro: Untyped
    CLOSE_TO_ZERO: Untyped
    def __init__(self, x, fun, jac, hess, hessp: Untyped | None = None, k_easy: float = 0.1, k_hard: float = 0.2): ...
    lambda_current: Untyped
    def solve(self, tr_radius) -> Untyped: ...
