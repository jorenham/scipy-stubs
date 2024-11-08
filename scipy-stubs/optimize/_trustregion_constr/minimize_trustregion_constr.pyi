from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy._typing import Untyped, UntypedCallable
from scipy.optimize._optimize import OptimizeResult as _OptimizeResult
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

###

TERMINATION_MESSAGES: dict[Literal[0, 1, 2, 3], str]  # undocumented

class OptimizeResult(_OptimizeResult):
    x: onpt.Array[tuple[int], np.float64]
    optimality: float | np.float64
    const_violation: float | np.float64
    fun: float | np.float64
    grad: onpt.Array[tuple[int], np.float64]
    lagrangian_grad: onpt.Array[tuple[int], np.float64]
    nit: int | np.intp
    nfev: int | np.intp
    njev: int | np.intp
    nhev: int | np.intp
    cg_niter: int
    method: Literal["equality_constrained_sqp", "tr_interior_point"]
    constr: list[float | np.float64]
    jac: list[onpt.Array[tuple[int, int], np.float64] | spmatrix]
    v: list[npt.NDArray[np.float64]]
    constr_nfev: list[int]
    constr_njev: list[int]
    constr_nhev: list[int]
    tr_radius: float | np.float64
    constr_penalty: float | np.float64
    barrier_tolerance: float | np.float64
    barrier_parameter: float | np.float64
    execution_time: float | np.float64
    message: str
    status: Literal[0, 1, 2, 3]
    cg_stop_cond: Literal[0, 1, 2, 3, 4]

# undocumented
class HessianLinearOperator:
    n: Final[int]
    hessp: Final[UntypedCallable]

    def __init__(self, /, hessp: UntypedCallable, n: int) -> None: ...
    def __call__(self, /, x: Untyped, *args: object) -> LinearOperator: ...

# undocumented
class LagrangianHessian:
    n: Final[int]
    objective_hess: Final[UntypedCallable]
    constraints_hess: Final[UntypedCallable]

    def __init__(self, /, n: int, objective_hess: UntypedCallable, constraints_hess: UntypedCallable) -> None: ...
    def __call__(self, /, x: Untyped, v_eq: Untyped, v_ineq: Untyped) -> LinearOperator: ...

# undocumented
def update_state_sqp(
    state: Untyped,
    x: Untyped,
    last_iteration_failed: Untyped,
    objective: Untyped,
    prepared_constraints: Untyped,
    start_time: Untyped,
    tr_radius: Untyped,
    constr_penalty: Untyped,
    cg_info: Untyped,
) -> Untyped: ...

# undocumented
def update_state_ip(
    state: Untyped,
    x: Untyped,
    last_iteration_failed: Untyped,
    objective: Untyped,
    prepared_constraints: Untyped,
    start_time: Untyped,
    tr_radius: Untyped,
    constr_penalty: Untyped,
    cg_info: Untyped,
    barrier_parameter: Untyped,
    barrier_tolerance: Untyped,
) -> Untyped: ...

#
def _minimize_trustregion_constr(
    fun: UntypedCallable,
    x0: npt.ArrayLike,
    args: tuple[object, ...],
    grad: Untyped,
    hess: Untyped,
    hessp: Untyped,
    bounds: Untyped,
    constraints: Untyped,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    barrier_tol: float = 1e-8,
    sparse_jacobian: bool | None = None,
    callback: UntypedCallable | None = None,
    maxiter: int = 1000,
    verbose: Literal[0, 1, 2] = 0,
    finite_diff_rel_step: npt.ArrayLike | None = None,
    initial_constr_penalty: float = 1.0,
    initial_tr_radius: float = 1.0,
    initial_barrier_parameter: float = 0.1,
    initial_barrier_tolerance: float = 0.1,
    factorization_method: str | None = None,
    disp: bool = False,
) -> OptimizeResult: ...
