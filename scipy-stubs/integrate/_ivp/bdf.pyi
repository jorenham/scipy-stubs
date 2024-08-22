from .base import DenseOutput as DenseOutput, OdeSolver as OdeSolver
from .common import (
    EPS as EPS,
    norm as norm,
    num_jac as num_jac,
    select_initial_step as select_initial_step,
    validate_first_step as validate_first_step,
    validate_max_step as validate_max_step,
    validate_tol as validate_tol,
    warn_extraneous as warn_extraneous,
)
from scipy._typing import Untyped
from scipy.linalg import lu_factor as lu_factor, lu_solve as lu_solve
from scipy.optimize._numdiff import group_columns as group_columns
from scipy.sparse import csc_matrix as csc_matrix, eye as eye, issparse as issparse
from scipy.sparse.linalg import splu as splu

MAX_ORDER: int
NEWTON_MAXITER: int
MIN_FACTOR: float
MAX_FACTOR: int

def compute_R(order, factor) -> Untyped: ...
def change_D(D, order, factor): ...
def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol) -> Untyped: ...

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
        fun,
        t0,
        y0,
        t_bound,
        max_step=...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        jac: Untyped | None = None,
        jac_sparsity: Untyped | None = None,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous,
    ): ...

class BdfDenseOutput(DenseOutput):
    order: Untyped
    t_shift: Untyped
    denom: Untyped
    D: Untyped
    def __init__(self, t_old, t, h, order, D) -> None: ...
