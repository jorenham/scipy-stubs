from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as OptimizeResult

def regularized_lsq_with_qr(
    m: Untyped,
    n: Untyped,
    R: Untyped,
    QTb: Untyped,
    perm: Untyped,
    diag: Untyped,
    copy_R: bool = True,
) -> Untyped: ...
def backtracking(
    A: Untyped,
    g: Untyped,
    x: Untyped,
    p: Untyped,
    theta: Untyped,
    p_dot_g: Untyped,
    lb: Untyped,
    ub: Untyped,
) -> Untyped: ...
def select_step(
    x: Untyped,
    A_h: Untyped,
    g_h: Untyped,
    c_h: Untyped,
    p: Untyped,
    p_h: Untyped,
    d: Untyped,
    lb: Untyped,
    ub: Untyped,
    theta: Untyped,
) -> Untyped: ...
def trf_linear(
    A: Untyped,
    b: Untyped,
    x_lsq: Untyped,
    lb: Untyped,
    ub: Untyped,
    tol: Untyped,
    lsq_solver: Untyped,
    lsmr_tol: Untyped,
    max_iter: Untyped,
    verbose: Untyped,
    *,
    lsmr_maxiter: Untyped | None = None,
) -> OptimizeResult: ...
