from typing_extensions import override

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped, UntypedCallable, UntypedTuple
from ._trustregion import BaseQuadraticSubproblem

__all__ = ["IterativeSubproblem", "_minimize_trustregion_exact", "estimate_smallest_singular_value", "singular_leading_submatrix"]

def _minimize_trustregion_exact(
    fun: UntypedCallable,
    x0: Untyped,
    args: UntypedTuple = (),
    jac: Untyped | None = None,
    hess: Untyped | None = None,
    **trust_region_options: Untyped,
) -> Untyped: ...
def estimate_smallest_singular_value(U: Untyped) -> Untyped: ...
def gershgorin_bounds(H: Untyped) -> Untyped: ...
def singular_leading_submatrix(A: Untyped, U: Untyped, k: Untyped) -> Untyped: ...

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
    lambda_current: Untyped
    def __init__(
        self,
        x: Untyped,
        fun: UntypedCallable,
        jac: Untyped,
        hess: Untyped,
        hessp: Untyped | None = None,
        k_easy: float = 0.1,
        k_hard: float = 0.2,
    ) -> None: ...
    @override
    def solve(self, trust_radius: float | np.float64) -> tuple[npt.NDArray[np.float64], bool]: ...
