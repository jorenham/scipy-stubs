from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as OptimizeResult
from scipy.sparse.linalg import LinearOperator as LinearOperator, aslinearoperator as aslinearoperator, lsmr as lsmr
from .common import (
    build_quadratic_1d as build_quadratic_1d,
    check_termination as check_termination,
    compute_grad as compute_grad,
    compute_jac_scale as compute_jac_scale,
    evaluate_quadratic as evaluate_quadratic,
    in_bounds as in_bounds,
    minimize_quadratic_1d as minimize_quadratic_1d,
    print_header_nonlinear as print_header_nonlinear,
    print_iteration_nonlinear as print_iteration_nonlinear,
    scale_for_robust_loss_function as scale_for_robust_loss_function,
    step_size_to_bound as step_size_to_bound,
    update_tr_radius as update_tr_radius,
)

def lsmr_operator(Jop, d, active_set) -> Untyped: ...
def find_intersection(x, tr_bounds, lb, ub) -> Untyped: ...
def dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub) -> Untyped: ...
def dogbox(
    fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose
) -> Untyped: ...
