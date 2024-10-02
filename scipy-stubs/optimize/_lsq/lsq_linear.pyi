from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import optype.numpy as onpt
from scipy.optimize import OptimizeResult
from scipy.optimize._typing import Bound
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_ScalarB1: TypeAlias = bool | np.bool_
_ScalarF8: TypeAlias = float | np.float64
_VectorF8: TypeAlias = onpt.Array[tuple[int], np.float64]

_ScalarInt_co: TypeAlias = np.integer[Any]
_ScalarFloat_co: TypeAlias = np.floating[Any] | _ScalarInt_co

_ScalarLikeFloat_co: TypeAlias = float | _ScalarFloat_co
_VectorLikeFloat_co: TypeAlias = Sequence[_ScalarLikeFloat_co] | onpt.CanArray[tuple[int], np.dtype[_ScalarFloat_co]]
_MatrixLikeFloat_co: TypeAlias = Sequence[_VectorLikeFloat_co] | onpt.CanArray[tuple[int, int], np.dtype[_ScalarFloat_co]]

_SparseArray: TypeAlias = sparray | spmatrix

TERMINATION_MESSAGES: Final[dict[int, str]]

def prepare_bounds(bounds: Bound, n: int) -> tuple[_ScalarF8, _ScalarF8] | tuple[_VectorF8, _VectorF8]: ...
def lsq_linear(
    A: LinearOperator | _SparseArray | _MatrixLikeFloat_co,
    b: _VectorLikeFloat_co,
    bounds: Bound = ...,
    method: Literal["trf", "bvls"] = "trf",
    tol: float = 1e-10,
    lsq_solver: Literal["exact", "lsmr"] | None = None,
    lsmr_tol: Literal["auto"] | float | None = None,
    max_iter: int | None = None,
    verbose: Literal[0, 1] | _ScalarB1 = 0,
    *,
    lsmr_maxiter: int | None = None,
) -> OptimizeResult: ...
