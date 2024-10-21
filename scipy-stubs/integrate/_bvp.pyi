from collections.abc import Callable
from typing import Any, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._typing import AnyReal
from scipy.interpolate import PPoly
from scipy.sparse import csc_matrix

_SCT = TypeVar("_SCT", bound=np.generic, default=np.float64)
_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any], default=np.float64 | np.complex128)

_Array_1d: TypeAlias = onpt.Array[tuple[int], _SCT]
_Array_2d: TypeAlias = onpt.Array[tuple[int, int], _SCT]
_Array_3d: TypeAlias = onpt.Array[tuple[int, int, int], _SCT]

_FunRHS: TypeAlias = Callable[[_Array_1d, _Array_2d[_SCT_fc]], npt.NDArray[_SCT_fc]]
_FunRHS_p: TypeAlias = Callable[[_Array_1d, _Array_2d[_SCT_fc], _Array_1d], npt.NDArray[_SCT_fc]]
_FunRHS_x: TypeAlias = Callable[[_Array_1d, _Array_2d[_SCT_fc], _Array_1d], _Array_2d[_SCT_fc]]

_FunBCR: TypeAlias = Callable[[_Array_1d[_SCT_fc], _Array_1d[_SCT_fc]], npt.NDArray[_SCT_fc]]
_FunBCR_p: TypeAlias = Callable[[_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d], npt.NDArray[_SCT_fc]]
_FunBCR_x: TypeAlias = Callable[[_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d], _Array_1d[_SCT_fc]]

_FunRHS_jac: TypeAlias = Callable[
    [_Array_1d, _Array_2d[_SCT_fc]],
    npt.NDArray[_SCT_fc],
]
_FunRHS_jac_p: TypeAlias = Callable[
    [_Array_1d, _Array_2d[_SCT_fc], _Array_1d],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]
_FunRHS_jac_x: TypeAlias = Callable[
    [_Array_1d, _Array_2d[_SCT_fc], _Array_1d],
    tuple[_Array_3d[_SCT_fc], _Array_3d[_SCT_fc] | None],
]

_FunBCR_jac: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc]],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]
_FunBCR_jac_p: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]
_FunBCR_jac_x: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d],
    tuple[_Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc] | None],
]

_FunCol: TypeAlias = Callable[
    [_Array_2d[_SCT_fc], _Array_1d],
    tuple[_Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc]],
]
_FunCol_jac: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_1d[_SCT_fc]],
    csc_matrix,
]

###

EPS: Final[float] = ...
TERMINATION_MESSAGES: Final[dict[Literal[0, 1, 2, 3], str]] = ...

# NOTE: this inherits from `scipy.optimize.OptimizeResult` at runtime.
# But because `BVPResult` doesn't share all members (and optional attributes
# still aren't a thing), it was omitted as a base class here.
class BVPResult(Generic[_SCT_fc]):
    sol: Final[PPoly]
    p: Final[_Array_1d | None]
    x: Final[_Array_1d]
    rms_residuals: Final[_Array_1d]
    niter: Final[int]
    status: Final[Literal[0, 1, 2]]
    message: Final[str]
    success: Final[bool]

    y: _Array_2d[_SCT_fc]
    yp: _Array_2d[_SCT_fc]

def estimate_fun_jac(
    fun: _FunRHS_x[_SCT_fc],
    x: _Array_1d,
    y: _Array_2d[_SCT_fc],
    p: _Array_1d,
    f0: _Array_2d[_SCT_fc] | None = None,
) -> tuple[_Array_3d[_SCT_fc], _Array_3d[_SCT_fc] | None]: ...  # undocumented
def estimate_bc_jac(
    bc: _FunBCR_x[_SCT_fc],
    ya: _Array_1d[_SCT_fc],
    yb: _Array_1d[_SCT_fc],
    p: _Array_1d,
    bc0: _Array_1d[_SCT_fc] | None = None,
) -> tuple[_Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc] | None]: ...  # undocumented
def compute_jac_indices(n: int, m: int, k: int) -> tuple[_Array_1d[np.intp], _Array_1d[np.intp]]: ...  # undocumented
def stacked_matmul(a: npt.NDArray[_SCT_fc], b: npt.NDArray[_SCT_fc]) -> npt.NDArray[_SCT_fc]: ...  # undocumented
def construct_global_jac(
    n: int,
    m: int,
    k: int,
    i_jac: _Array_1d[np.intp],
    j_jac: _Array_1d[np.intp],
    h: float,
    df_dy: _Array_3d[_SCT_fc],
    df_dy_middle: _Array_3d[_SCT_fc],
    df_dp: _Array_3d[_SCT_fc] | None,
    df_dp_middle: _Array_3d[_SCT_fc] | None,
    dbc_dya: _Array_2d[_SCT_fc],
    dbc_dyb: _Array_2d[_SCT_fc],
    dbc_dp: _Array_2d[_SCT_fc] | None,
) -> csc_matrix: ...  # undocumented
def collocation_fun(
    fun: _FunRHS_x[_SCT_fc],
    y: _Array_2d[_SCT_fc],
    p: _Array_1d,
    x: _Array_1d,
    h: float,
) -> tuple[_Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc], _Array_2d[_SCT_fc]]: ...  # undocumented
def prepare_sys(
    n: int,
    m: int,
    k: int,
    fun: _FunRHS_x[_SCT_fc],
    bc: _FunBCR_x[_SCT_fc],
    fun_jac: _FunRHS_jac_x[_SCT_fc] | None,
    bc_jac: _FunBCR_jac_x[_SCT_fc] | None,
    x: _Array_1d,
    h: float,
) -> tuple[_FunCol[_SCT_fc], _FunCol_jac[_SCT_fc]]: ...  # undocumented
def solve_newton(
    n: int,
    m: int,
    h: int,
    col_fun: _FunCol[_SCT_fc],
    bc: _FunBCR_x[_SCT_fc],
    jac: _FunCol_jac[_SCT_fc],
    y: _Array_2d[_SCT_fc],
    p: _Array_1d,
    B: _Array_2d | None,
    bvp_tol: float,
    bc_tol: float,
) -> tuple[_Array_2d[_SCT_fc], _Array_1d, bool]: ...  # undocumented
def print_iteration_header() -> None: ...  # undocumented
def print_iteration_progress(
    iteration: int,
    residual: complex,
    bc_residual: complex,
    total_nodes: int,
    nodes_added: int,
) -> None: ...  # undocumented
def estimate_rms_residuals(
    fun: _FunRHS_x[_SCT_fc],
    sol: PPoly,
    x: _Array_1d,
    h: float,
    p: _Array_1d,
    r_middle: _Array_2d[_SCT_fc],
    f_middle: _Array_2d[_SCT_fc],
) -> _Array_1d: ...  # undocumented
def create_spline(y: _Array_2d[_SCT_fc], yp: _Array_2d[_SCT_fc], x: _Array_1d, h: float) -> PPoly: ...  # undocumented
def modify_mesh(x: _Array_1d, insert_1: _Array_1d[np.intp], insert_2: _Array_1d[np.intp]) -> _Array_1d: ...  # undocumented
@overload
def wrap_functions(
    fun: _FunRHS[_SCT_fc],
    bc: _FunBCR[_SCT_fc],
    fun_jac: _FunRHS_jac[_SCT_fc] | None,
    bc_jac: _FunBCR_jac[_SCT_fc] | None,
    k: Literal[False, 0],
    a: AnyReal,
    S: _Array_2d | None,
    D: _Array_2d | None,
    dtype: type[float | complex],
) -> tuple[_FunRHS_x[_SCT_fc], _FunBCR_x[_SCT_fc], _FunRHS_jac_x[_SCT_fc], _FunBCR_jac_x[_SCT_fc]]: ...  # undocumented
@overload
def wrap_functions(
    fun: _FunRHS_p[_SCT_fc],
    bc: _FunBCR_p[_SCT_fc],
    fun_jac: _FunRHS_jac_p[_SCT_fc] | None,
    bc_jac: _FunBCR_jac_p[_SCT_fc] | None,
    k: Literal[True, 1],
    a: AnyReal,
    S: _Array_2d | None,
    D: _Array_2d | None,
    dtype: type[float | complex],
) -> tuple[_FunRHS_x[_SCT_fc], _FunBCR_x[_SCT_fc], _FunRHS_jac_x[_SCT_fc], _FunBCR_jac_x[_SCT_fc]]: ...  # undocumented

#

@overload
def solve_bvp(
    fun: _FunRHS[_SCT_fc],
    bc: _FunBCR[_SCT_fc],
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeNumber_co,
    p: None = None,
    S: _ArrayLikeFloat_co | None = None,
    fun_jac: _FunRHS_jac[_SCT_fc] | None = None,
    bc_jac: _FunBCR_jac[_SCT_fc] | None = None,
    tol: float = 0.001,
    max_nodes: int = 1_000,
    verbose: Literal[0, 1, 2] = 0,
    bc_tol: float | None = None,
) -> BVPResult[_SCT_fc]: ...
@overload
def solve_bvp(
    fun: _FunRHS_p[_SCT_fc],
    bc: _FunBCR_p[_SCT_fc],
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeNumber_co,
    p: _ArrayLikeFloat_co,
    S: _ArrayLikeFloat_co | None = None,
    fun_jac: _FunRHS_jac_p[_SCT_fc] | None = None,
    bc_jac: _FunBCR_jac_p[_SCT_fc] | None = None,
    tol: float = 0.001,
    max_nodes: int = 1_000,
    verbose: Literal[0, 1, 2] = 0,
    bc_tol: float | None = None,
) -> BVPResult[_SCT_fc]: ...
