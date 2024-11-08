from typing import Final, Literal, TypeAlias, type_check_only

import numpy as np
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyInt, AnyReal, Untyped
from scipy.optimize import OptimizeResult
from scipy.optimize._typing import Bound
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

_Scalar_f8: TypeAlias = float | np.float64
_Vector_i0: TypeAlias = onpt.Array[tuple[int], np.intp]
_Vector_f8: TypeAlias = onpt.Array[tuple[int], np.float64]

_BoundsLike: TypeAlias = tuple[_ArrayLikeFloat_co, _ArrayLikeFloat_co] | Bound
_TerminationStatus: TypeAlias = Literal[-1, 0, 1, 2, 3]

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: _Vector_f8
    fun: _Vector_f8
    const: _Scalar_f8
    optimality: _Scalar_f8
    active_mask: _Vector_i0
    unbounded_sol: tuple[Untyped, ...]
    nit: int
    status: _TerminationStatus
    message: str
    success: bool

###

TERMINATION_MESSAGES: Final[dict[_TerminationStatus, str]] = ...

def lsq_linear(
    A: _ArrayLikeFloat_co | _spbase | LinearOperator,
    b: _ArrayLikeFloat_co,
    bounds: _BoundsLike = ...,
    method: Literal["trf", "bvls"] = "trf",
    tol: AnyReal = 1e-10,
    lsq_solver: Literal["exact", "lsmr"] | None = None,
    lsmr_tol: AnyReal | Literal["auto"] | None = None,
    max_iter: AnyInt | None = None,
    verbose: Literal[0, 1, 2] = 0,
    *,
    lsmr_maxiter: AnyInt | None = None,
) -> _OptimizeResult: ...

# undocumented
def prepare_bounds(bounds: _BoundsLike, n: op.CanIndex) -> tuple[_Scalar_f8, _Scalar_f8] | tuple[_Vector_f8, _Vector_f8]: ...
