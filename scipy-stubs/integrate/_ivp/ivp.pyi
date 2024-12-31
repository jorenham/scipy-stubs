from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Final, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypedDict, TypeVar, Unpack

import numpy as np
import optype.numpy as onp
from scipy._lib._util import _RichResult
from scipy._typing import Falsy, Truthy
from scipy.sparse import sparray, spmatrix
from .base import DenseOutput, OdeSolver
from .common import OdeSolution

_SCT_cf = TypeVar("_SCT_cf", bound=np.inexact[Any], default=np.float64 | np.complex128)

_FuncSol: TypeAlias = Callable[[float], onp.ArrayND[_SCT_cf]]
_FuncEvent: TypeAlias = Callable[[float, onp.ArrayND[_SCT_cf]], float]
_Events: TypeAlias = Sequence[_FuncEvent[_SCT_cf]]

_Int1D: TypeAlias = onp.Array1D[np.intp]
_Float1D: TypeAlias = onp.Array1D[np.float64]

_ToJac: TypeAlias = onp.ToComplex2D | spmatrix | sparray

_IVPMethod: TypeAlias = Literal["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"]

@type_check_only
class _SolverOptions(TypedDict, Generic[_SCT_cf], total=False):
    first_step: onp.ToFloat | None
    max_step: onp.ToFloat
    rtol: onp.ToFloat | onp.ToFloat1D
    atol: onp.ToFloat | onp.ToFloat1D
    jac: _ToJac | Callable[[float, onp.Array1D[np.float64]], _ToJac] | None
    jac_sparsity: onp.ToFloat2D | spmatrix | sparray | None
    lband: onp.ToInt | None
    uband: onp.ToInt | None
    min_step: onp.ToFloat

###

METHODS: Final[dict[str, type]] = ...
MESSAGES: Final[dict[int, str]] = ...

class OdeResult(
    _RichResult[int | str | onp.ArrayND[np.float64 | _SCT_cf] | list[onp.ArrayND[np.float64 | _SCT_cf]] | OdeSolution | None],
    Generic[_SCT_cf],
):
    t: _Float1D
    y: onp.Array2D[_SCT_cf]
    sol: OdeSolution | None
    t_events: list[_Float1D] | None
    y_events: list[onp.ArrayND[_SCT_cf]] | None
    nfev: int
    njev: int
    nlu: int
    status: Literal[-1, 0, 1]
    message: str
    success: bool

def prepare_events(events: _FuncEvent[_SCT_cf] | _Events[_SCT_cf]) -> tuple[_Events[_SCT_cf], _Float1D, _Float1D]: ...
def solve_event_equation(event: _FuncEvent[_SCT_cf], sol: _FuncSol[_SCT_cf], t_old: float, t: float) -> float: ...
def handle_events(
    sol: DenseOutput,
    events: Sequence[_FuncEvent[_SCT_cf]],
    active_events: onp.ArrayND[np.intp],
    event_count: onp.ArrayND[np.intp | np.float64],
    max_events: onp.ArrayND[np.intp | np.float64],
    t_old: float,
    t: float,
) -> tuple[_Int1D, _Float1D, bool]: ...
def find_active_events(g: onp.ToFloat1D, g_new: onp.ToFloat1D, direction: onp.ArrayND[np.float64]) -> _Int1D: ...

#
@overload
def solve_ivp(
    fun: Callable[Concatenate[float, onp.Array1D[_SCT_cf], ...], onp.ArrayND[_SCT_cf]],
    t_span: Sequence[onp.ToFloat],
    y0: onp.ToArray1D,
    method: _IVPMethod | type[OdeSolver] = "RK45",
    t_eval: onp.ToFloat1D | None = None,
    dense_output: bool = False,
    events: _Events[_SCT_cf] | None = None,
    vectorized: Falsy = False,
    args: tuple[object, ...] | None = None,
    **options: Unpack[_SolverOptions],
) -> OdeResult[_SCT_cf]: ...
@overload
def solve_ivp(
    fun: Callable[Concatenate[_Float1D, onp.Array2D[_SCT_cf], ...], onp.ArrayND[_SCT_cf]],
    t_span: Sequence[onp.ToFloat],
    y0: onp.ToArray1D,
    method: _IVPMethod | type[OdeSolver] = "RK45",
    t_eval: onp.ToFloat1D | None = None,
    dense_output: bool = False,
    events: _Events[_SCT_cf] | None = None,
    *,
    vectorized: Truthy,
    args: tuple[object, ...] | None = None,
    **options: Unpack[_SolverOptions],
) -> OdeResult[_SCT_cf]: ...
