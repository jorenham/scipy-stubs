from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy._typing import UntypedCallable
from scipy.interpolate import PPoly

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any], default=np.float64 | np.complex128)
_Array_1d: TypeAlias = onpt.Array[tuple[int], _SCT]
_Array_2d: TypeAlias = onpt.Array[tuple[int, int], _SCT]

_FunRHS: TypeAlias = Callable[
    [_Array_1d[np.float64], _Array_2d[_SCT_fc]],
    npt.NDArray[_SCT_fc],
]
_FunRHS_p: TypeAlias = Callable[
    [_Array_1d[np.float64], _Array_2d[_SCT_fc], _Array_1d[np.float64]],
    npt.NDArray[_SCT_fc],
]
_FunRHS_jac: TypeAlias = Callable[
    [_Array_1d[np.float64], _Array_2d[_SCT_fc]],
    npt.NDArray[_SCT_fc],
]
_FunRHS_jac_p: TypeAlias = Callable[
    [_Array_1d[np.float64], _Array_2d[_SCT_fc], _Array_1d[np.float64]],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]
_FunBCR: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc]],
    npt.NDArray[_SCT_fc],
]
_FunBCR_p: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d[np.float64]],
    npt.NDArray[_SCT_fc],
]
_FunBCR_jac: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc]],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]
_FunBCR_jac_p: TypeAlias = Callable[
    [_Array_1d[_SCT_fc], _Array_1d[_SCT_fc], _Array_1d[np.float64]],
    tuple[npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc], npt.NDArray[_SCT_fc]],
]

EPS: Final[float]
TERMINATION_MESSAGES: Final[dict[int, str]]

# private functions
# TODO: Annotate these
estimate_fun_jac: UntypedCallable
estimate_bc_jac: UntypedCallable
compute_jac_indices: UntypedCallable
stacked_matmul: UntypedCallable
construct_global_jac: UntypedCallable
collocation_fun: UntypedCallable
prepare_sys: UntypedCallable
solve_newton: UntypedCallable
print_iteration_progress: Callable[..., None]
estimate_rms_residuals: UntypedCallable
create_spline: UntypedCallable
modify_mesh: UntypedCallable
wrap_functions: UntypedCallable

def print_iteration_header() -> None: ...

# NOTE: this inherits from `scipy.optimize.OptimizeResult` at runtime.
# But because `BVPResult` doesn't share all members (and optional attributes
# still aren't a thing), it was omitted as a base class here.
class BVPResult(Generic[_SCT_fc]):
    sol: Final[PPoly]
    p: Final[onpt.Array[tuple[int], np.float64] | None]
    x: Final[onpt.Array[tuple[int], np.float64]]
    rms_residuals: Final[onpt.Array[tuple[int], np.float64]]
    niter: Final[int]
    status: Final[Literal[0, 1, 2]]
    message: Final[str]
    success: Final[bool]

    y: onpt.Array[tuple[int, int], _SCT_fc]
    yp: onpt.Array[tuple[int, int], _SCT_fc]

# public
@overload
def solve_bvp(
    fun: _FunRHS[_SCT_fc],
    bc: _FunBCR[_SCT_fc],
    x: onpt.AnyFloatingArray | Sequence[float],
    y: onpt.AnyInexactArray | Sequence[Sequence[complex]],
    p: None = None,
    S: onpt.AnyFloatingArray | Sequence[Sequence[float]] | None = None,
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
    x: onpt.AnyFloatingArray | Sequence[float],
    y: onpt.AnyInexactArray | Sequence[Sequence[complex]],
    p: onpt.AnyFloatingArray | Sequence[float],
    S: onpt.AnyFloatingArray | Sequence[Sequence[float]] | None = None,
    fun_jac: _FunRHS_jac_p[_SCT_fc] | None = None,
    bc_jac: _FunBCR_jac_p[_SCT_fc] | None = None,
    tol: float = 0.001,
    max_nodes: int = 1_000,
    verbose: Literal[0, 1, 2] = 0,
    bc_tol: float | None = None,
) -> BVPResult[_SCT_fc]: ...
