from typing import Any, Literal, TypeAlias, type_check_only

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult

_Scalar_i: TypeAlias = int | np.integer[Any]
_Scalar_f8: TypeAlias = float | np.float64
_Vector_f8: TypeAlias = onp.Array1D[np.float64]

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: _Vector_f8
    fun: _Scalar_f8
    cost: Untyped
    optimality: Untyped
    active_mask: Untyped
    nit: int
    status: int
    initial_cost: Untyped

###

def regularized_lsq_with_qr(
    m: int,
    n: int,
    R: onp.ArrayND[np.number[Any]],
    QTb: onp.ArrayND[np.number[Any]],
    perm: onp.ArrayND[np.number[Any]],
    diag: onp.ArrayND[np.number[Any]],
    copy_R: bool = True,
) -> _Vector_f8: ...

# undocumented
def backtracking(
    A: onp.ArrayND[np.number[Any]],
    g: Untyped,
    x: Untyped,
    p: Untyped,
    theta: Untyped,
    p_dot_g: Untyped,
    lb: onp.ArrayND[np.number[Any]],
    ub: onp.ArrayND[np.number[Any]],
) -> Untyped: ...

# undocumented
def select_step(
    x: Untyped,
    A_h: Untyped,
    g_h: Untyped,
    c_h: Untyped,
    p: Untyped,
    p_h: Untyped,
    d: Untyped,
    lb: onp.ArrayND[np.number[Any]],
    ub: onp.ArrayND[np.number[Any]],
    theta: Untyped,
) -> Untyped: ...

# undocumented
def trf_linear(
    A: onp.ArrayND[np.number[Any]],
    b: onp.ArrayND[np.number[Any]],
    x_lsq: onp.ArrayND[np.number[Any]],
    lb: onp.ArrayND[np.number[Any]],
    ub: onp.ArrayND[np.number[Any]],
    tol: _Scalar_f8,
    lsq_solver: Literal["exact", "lsmr"],
    lsmr_tol: _Scalar_f8 | None,
    max_iter: _Scalar_i,
    verbose: Literal[0, 1, 2],
    *,
    lsmr_maxiter: _Scalar_i | None = None,
) -> _OptimizeResult: ...
