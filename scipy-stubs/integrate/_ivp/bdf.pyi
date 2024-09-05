# TODO: Finish this

from typing import Final

from scipy._typing import Untyped, UntypedCallable
from .base import DenseOutput, OdeSolver

MAX_ORDER: Final = 5
NEWTON_MAXITER: Final = 4
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

def compute_R(order: Untyped, factor: Untyped) -> Untyped: ...
def change_D(D: Untyped, order: Untyped, factor: Untyped) -> Untyped: ...
def solve_bdf_system(
    fun: UntypedCallable,
    t_new: Untyped,
    y_predict: Untyped,
    c: Untyped,
    psi: Untyped,
    LU: Untyped,
    solve_lu: Untyped,
    scale: Untyped,
    tol: Untyped,
) -> Untyped: ...

class BDF(OdeSolver):
    max_step: Untyped
    h_abs: Untyped
    h_abs_old: Untyped
    error_norm_old: Untyped
    newton_tol: Untyped
    jac_factor: Untyped
    lu: Untyped
    solve_lu: Untyped
    I: Untyped
    gamma: Untyped
    alpha: Untyped
    error_const: Untyped
    D: Untyped
    order: int
    n_equal_steps: int
    LU: Untyped
    def __init__(
        self,
        /,
        fun: UntypedCallable,
        t0: Untyped,
        y0: Untyped,
        t_bound: Untyped,
        max_step: Untyped = ...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        jac: Untyped | None = None,
        jac_sparsity: Untyped | None = None,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous: Untyped,
    ) -> None: ...

class BdfDenseOutput(DenseOutput):
    order: Untyped
    t_shift: Untyped
    denom: Untyped
    D: Untyped
    def __init__(self, /, t_old: float, t: float, h: Untyped, order: Untyped, D: Untyped) -> None: ...
