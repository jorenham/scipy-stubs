from collections.abc import Callable
from typing import Any, Final, Generic, TypeAlias
from typing_extensions import Never, TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._typing import AnyReal
from scipy.sparse import sparray, spmatrix
from .base import DenseOutput, OdeSolver

_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.inexact[Any], default=np.float64 | np.complex128)

# TODO(jorenham): sparse
_LU: TypeAlias = tuple[npt.NDArray[np.inexact[Any]], npt.NDArray[np.integer[Any]]]
_FuncLU: TypeAlias = Callable[[_ArrayLikeNumber_co], _LU]
_FuncSolveLU: TypeAlias = Callable[[_LU, npt.ArrayLike], npt.NDArray[np.inexact[Any]]]

###

MAX_ORDER: Final = 5
NEWTON_MAXITER: Final = 4
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

class BDF(OdeSolver, Generic[_SCT_co]):
    max_step: float
    h_abs: float
    h_abs_old: float | None
    error_norm_old: None
    newton_tol: float
    jac_factor: npt.NDArray[np.float64] | None  # 1d

    LU: _LU
    lu: _FuncLU
    solve_lu: _FuncSolveLU

    I: npt.NDArray[_SCT_co]
    error_const: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    D: npt.NDArray[np.float64]
    order: int
    n_equal_steps: int

    def __init__(
        self,
        /,
        fun: Callable[[float, npt.NDArray[_SCT_co]], _ArrayLikeNumber_co],
        t0: AnyReal,
        y0: npt.NDArray[_SCT_co] | _ArrayLikeNumber_co,
        t_bound: AnyReal,
        max_step: AnyReal = ...,
        rtol: AnyReal = 0.001,
        atol: AnyReal = 1e-06,
        jac: (
            _ArrayLikeNumber_co
            | spmatrix
            | sparray
            | Callable[[float, npt.NDArray[_SCT_co]], _ArrayLikeNumber_co | spmatrix | sparray]
            | None
        ) = None,
        jac_sparsity: _ArrayLikeFloat_co | spmatrix | sparray | None = None,
        vectorized: bool = False,
        first_step: AnyReal | None = None,
        **extraneous: Never,
    ) -> None: ...

class BdfDenseOutput(DenseOutput):
    order: int
    t_shift: npt.NDArray[np.float64]
    denom: npt.NDArray[np.float64]
    D: npt.NDArray[np.float64]
    def __init__(self, /, t_old: float, t: float, h: float, order: int, D: npt.NDArray[np.float64]) -> None: ...

def compute_R(order: int, factor: float) -> npt.NDArray[np.float64]: ...
def change_D(D: npt.NDArray[np.float64], order: int, factor: float) -> None: ...
def solve_bdf_system(
    fun: Callable[[float, npt.NDArray[_SCT_co]], _ArrayLikeNumber_co],
    t_new: AnyReal,
    y_predict: npt.NDArray[_SCT_co],
    c: float,
    psi: npt.NDArray[np.float64],
    LU: _FuncLU,
    solve_lu: _FuncSolveLU,
    scale: npt.NDArray[np.float64],
    tol: float,
) -> tuple[bool, int, npt.NDArray[_SCT_co], npt.NDArray[_SCT_co]]: ...
