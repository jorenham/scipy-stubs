from collections.abc import Callable
from typing import Final, Literal, TypeAlias
from typing_extensions import Never

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co
from scipy.sparse import sparray, spmatrix
from .base import DenseOutput, OdeSolver

_LU: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]
_FuncSolveLU: TypeAlias = Callable[[_LU, npt.NDArray[np.float64]], npt.NDArray[np.float64]]

###

S6: Final[float] = ...

C: Final[npt.NDArray[np.float64]] = ...
E: Final[npt.NDArray[np.float64]] = ...

MU_REAL: Final[float] = ...
MU_COMPLEX: Final[complex] = ...

T: Final[npt.NDArray[np.float64]] = ...
TI: Final[npt.NDArray[np.float64]] = ...
TI_REAL: Final[npt.NDArray[np.float64]] = ...
TI_COMPLEX: Final[npt.NDArray[np.complex128]] = ...

P: Final[npt.NDArray[np.float64]] = ...

NEWTON_MAXITER: Final = 6
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

class Radau(OdeSolver):
    max_step: float
    h_abs: float
    h_abs_old: float | None
    error_norm_old: float | None
    newton_tol: float
    jac_factor: npt.NDArray[np.float64] | None  # 1d
    current_jac: bool

    LU_real: _LU
    LU_complex: _LU
    lu: Callable[[npt.NDArray[np.float64]], _LU]
    solve_lu: _FuncSolveLU

    y_old: npt.NDArray[np.float64] | None
    f: npt.NDArray[np.float64]
    I: npt.NDArray[np.float64]
    Z: npt.NDArray[np.float64] | None
    sol: RadauDenseOutput | None

    def __init__(
        self,
        /,
        fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeFloat_co],
        t0: onp.ToFloat,
        y0: _ArrayLikeFloat_co,
        t_bound: onp.ToFloat,
        max_step: onp.ToFloat = ...,
        rtol: onp.ToFloat = 0.001,
        atol: onp.ToFloat = 1e-06,
        jac: (
            _ArrayLikeFloat_co
            | spmatrix
            | sparray
            | Callable[[float, npt.NDArray[np.float64]], _ArrayLikeFloat_co | spmatrix | sparray]
            | None
        ) = None,
        jac_sparsity: _ArrayLikeFloat_co | spmatrix | sparray | None = None,
        vectorized: bool = False,
        first_step: onp.ToFloat | None = None,
        **extraneous: Never,
    ) -> None: ...

class RadauDenseOutput(DenseOutput):
    order: int
    h: float
    Q: npt.NDArray[np.float64]
    y_old: npt.NDArray[np.float64]

    def __init__(self, /, t_old: float, t: float, y_old: npt.NDArray[np.float64], Q: npt.NDArray[np.float64]) -> None: ...

def solve_collocation_system(
    fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeFloat_co],
    t: onp.ToFloat,
    y: npt.NDArray[np.float64],
    h: onp.ToFloat,
    Z0: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    tol: onp.ToFloat,
    LU_real: _LU,
    LU_complex: _LU,
    solve_lu: _FuncSolveLU,
) -> tuple[bool, int, onp.Array[tuple[Literal[3], int], np.float64], float | None]: ...
def predict_factor(
    h_abs: onp.ToFloat,
    h_abs_old: onp.ToFloat,
    error_norm: onp.ToFloat,
    error_norm_old: onp.ToFloat,
) -> onp.ToFloat: ...
