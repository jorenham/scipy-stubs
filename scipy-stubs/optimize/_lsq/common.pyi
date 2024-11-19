from typing import Any, Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_SparseArray: TypeAlias = sparray | spmatrix
_FloatingND: TypeAlias = npt.NDArray[np.floating[Any]]

EPS: Final[float]

def intersect_trust_region(
    x: npt.ArrayLike,
    s: npt.ArrayLike,
    Delta: onp.ToFloat,
) -> tuple[float | np.float64, float | np.float64]: ...
def solve_lsq_trust_region(
    n: int,
    m: int,
    uf: _FloatingND,
    s: _FloatingND,
    V: _FloatingND,
    Delta: onp.ToFloat,
    initial_alpha: onp.ToFloat | None = None,
    rtol: onp.ToFloat = 0.01,
    max_iter: onp.ToInt = 10,
) -> tuple[onp.Array1D[np.float64], float, int]: ...
def solve_trust_region_2d(B: npt.ArrayLike, g: npt.ArrayLike, Delta: onp.ToFloat) -> tuple[onp.Array1D[np.float64], bool]: ...
def update_tr_radius(
    Delta: onp.ToFloat,
    actual_reduction: onp.ToFloat,
    predicted_reduction: onp.ToFloat,
    step_norm: onp.ToFloat,
    bound_hit: op.CanBool,
) -> tuple[float, float]: ...
def build_quadratic_1d(
    J: _FloatingND | _SparseArray | LinearOperator,
    g: _FloatingND,
    s: _FloatingND,
    diag: _FloatingND | None = None,
    s0: _FloatingND | None = None,
) -> tuple[float, float, float]: ...
def minimize_quadratic_1d(
    a: onp.ToFloat,
    b: onp.ToFloat,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    c: onp.ToFloat = 0,
) -> tuple[float, float]: ...
def evaluate_quadratic(
    J: _FloatingND | _SparseArray | LinearOperator,
    g: _FloatingND,
    s: _FloatingND,
    diag: _FloatingND | None = None,
) -> np.float64 | onp.Array1D[np.float64]: ...
def in_bounds(x: _FloatingND, lb: npt.ArrayLike, ub: npt.ArrayLike) -> np.bool_: ...
def step_size_to_bound(
    x: npt.ArrayLike,
    s: npt.ArrayLike,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.int_]]: ...
def find_active_constraints(
    x: npt.ArrayLike,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    rtol: onp.ToFloat = 1e-10,
) -> npt.NDArray[np.int_]: ...
def make_strictly_feasible(x: _FloatingND, lb: npt.ArrayLike, ub: npt.ArrayLike, rstep: onp.ToFloat = 1e-10) -> _FloatingND: ...
def CL_scaling_vector(
    x: _FloatingND,
    g: _FloatingND,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
) -> tuple[_FloatingND, _FloatingND]: ...
def reflective_transformation(y: _FloatingND, lb: npt.ArrayLike, ub: npt.ArrayLike) -> tuple[_FloatingND, _FloatingND]: ...
def print_header_nonlinear() -> None: ...
def print_iteration_nonlinear(
    iteration: int,
    nfev: int,
    cost: float,
    cost_reduction: float,
    step_norm: float,
    optimality: float,
) -> None: ...
def print_header_linear() -> None: ...
def print_iteration_linear(
    iteration: int,
    cost: float,
    cost_reduction: float,
    step_norm: float,
    optimality: float,
) -> None: ...
def compute_grad(J: _FloatingND | _SparseArray | LinearOperator, f: _FloatingND) -> _FloatingND | _SparseArray: ...
def compute_jac_scale(
    J: _FloatingND | _SparseArray | LinearOperator,
    scale_inv_old: _FloatingND | onp.ToFloat | None = None,
) -> tuple[_FloatingND, _FloatingND]: ...
def left_multiplied_operator(J: _FloatingND | _SparseArray | LinearOperator, d: _FloatingND) -> LinearOperator: ...
def right_multiplied_operator(J: _FloatingND | _SparseArray | LinearOperator, d: _FloatingND) -> LinearOperator: ...
def regularized_lsq_operator(J: _FloatingND | _SparseArray | LinearOperator, diag: _FloatingND) -> LinearOperator: ...
def right_multiply(
    J: _FloatingND | _SparseArray | LinearOperator,
    d: _FloatingND,
    copy: bool = True,
) -> _FloatingND | _SparseArray | LinearOperator: ...
def left_multiply(
    J: _FloatingND | _SparseArray | LinearOperator,
    d: _FloatingND,
    copy: bool = True,
) -> _FloatingND | _SparseArray | LinearOperator: ...
def check_termination(
    dF: onp.ToFloat,
    F: onp.ToFloat,
    dx_norm: onp.ToFloat,
    x_norm: onp.ToFloat,
    ratio: onp.ToFloat,
    ftol: onp.ToFloat,
    xtol: onp.ToFloat,
) -> Literal[2, 3, 4] | None: ...
def scale_for_robust_loss_function(
    J: _FloatingND | _SparseArray | LinearOperator,
    f: onp.ToFloat,
    rho: _FloatingND,
) -> tuple[_FloatingND | _SparseArray | LinearOperator, onp.ToFloat]: ...
