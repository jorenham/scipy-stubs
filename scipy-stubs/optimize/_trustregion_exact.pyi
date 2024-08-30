from typing_extensions import override

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped
from ._trustregion import BaseQuadraticSubproblem

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
    def __init__(self, x, fun, jac, hess, hessp: Untyped | None = None, k_easy: float = 0.1, k_hard: float = 0.2) -> None: ...
    lambda_current: Untyped
    @override
    def solve(self, trust_radius: float | np.float64) -> tuple[npt.NDArray[np.float64], bool]: ...
