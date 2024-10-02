from typing import Any, Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy._typing as spt
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_SparseArray: TypeAlias = sparray | spmatrix

EPS: Final[float]

def intersect_trust_region(
    x: npt.ArrayLike,
    s: npt.ArrayLike,
    Delta: spt.AnyReal,
) -> tuple[float | np.float64, float | np.float64]: ...
def solve_lsq_trust_region(
    n: int,
    m: int,
    uf: npt.NDArray[np.floating[Any]],
    s: npt.NDArray[np.floating[Any]],
    V: npt.NDArray[np.floating[Any]],
    Delta: spt.AnyReal,
    initial_alpha: spt.AnyReal | None = None,
    rtol: spt.AnyReal = 0.01,
    max_iter: spt.AnyInt = 10,
) -> tuple[onpt.Array[tuple[int], np.float64], float, int]: ...
def solve_trust_region_2d(
    B: npt.ArrayLike,
    g: npt.ArrayLike,
    Delta: spt.AnyReal,
) -> tuple[onpt.Array[tuple[Literal[2]], np.float64], bool]: ...
def update_tr_radius(
    Delta: spt.AnyReal,
    actual_reduction: spt.AnyReal,
    predicted_reduction: spt.AnyReal,
    step_norm: spt.AnyReal,
    bound_hit: spt.AnyBool,
) -> tuple[float, float]: ...
def build_quadratic_1d(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    g: npt.NDArray[np.floating[Any]],
    s: npt.NDArray[np.floating[Any]],
    diag: npt.NDArray[np.floating[Any]] | None = None,
    s0: npt.NDArray[np.floating[Any]] | None = None,
) -> tuple[float, float, float]: ...
def minimize_quadratic_1d(
    a: spt.AnyReal,
    b: spt.AnyReal,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    c: spt.AnyReal = 0,
) -> tuple[float, float]: ...
def evaluate_quadratic(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    g: npt.NDArray[np.floating[Any]],
    s: npt.NDArray[np.floating[Any]],
    diag: npt.NDArray[np.floating[Any]] | None = None,
) -> np.float64 | onpt.Array[tuple[int], np.float64]: ...
def in_bounds(x: npt.NDArray[np.floating[Any]], lb: npt.ArrayLike, ub: npt.ArrayLike) -> np.bool_: ...
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
    rtol: spt.AnyReal = 1e-10,
) -> npt.NDArray[np.int_]: ...
def make_strictly_feasible(
    x: npt.NDArray[np.floating[Any]],
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    rstep: spt.AnyReal = 1e-10,
) -> npt.NDArray[np.floating[Any]]: ...
def CL_scaling_vector(
    x: npt.NDArray[np.floating[Any]],
    g: npt.NDArray[np.floating[Any]],
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]: ...
def reflective_transformation(
    y: npt.NDArray[np.floating[Any]],
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]: ...
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
def compute_grad(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    f: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]] | _SparseArray: ...
def compute_jac_scale(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    scale_inv_old: npt.NDArray[np.floating[Any]] | spt.AnyReal | None = None,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]: ...
def left_multiplied_operator(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    d: npt.NDArray[np.floating[Any]],
) -> LinearOperator: ...
def right_multiplied_operator(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    d: npt.NDArray[np.floating[Any]],
) -> LinearOperator: ...
def regularized_lsq_operator(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    diag: npt.NDArray[np.floating[Any]],
) -> LinearOperator: ...
def right_multiply(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    d: npt.NDArray[np.floating[Any]],
    copy: bool = True,
) -> npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator: ...
def left_multiply(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    d: npt.NDArray[np.floating[Any]],
    copy: bool = True,
) -> npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator: ...
def check_termination(
    dF: spt.AnyReal,
    F: spt.AnyReal,
    dx_norm: spt.AnyReal,
    x_norm: spt.AnyReal,
    ratio: spt.AnyReal,
    ftol: spt.AnyReal,
    xtol: spt.AnyReal,
) -> Literal[2, 3, 4] | None: ...
def scale_for_robust_loss_function(
    J: npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator,
    f: spt.AnyReal,
    rho: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.floating[Any]] | _SparseArray | LinearOperator, spt.AnyReal]: ...
