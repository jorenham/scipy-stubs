# TODO: Finish this

from collections.abc import Sequence
from typing import Any, Final, Generic, Literal
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
from scipy._typing import Untyped, UntypedArray, UntypedCallable, UntypedTuple
from .base import DenseOutput, OdeSolver
from .common import OdeSolution

_SCT_cf = TypeVar("_SCT_cf", bound=np.inexact[Any], default=np.float64 | np.complex128)

METHODS: Final[dict[str, type]]
MESSAGES: Final[dict[int, str]]

# TODO
class OdeResult(Generic[_SCT_cf]):
    t: onpt.Array[tuple[int], np.float64]
    y: onpt.Array[tuple[int, int], _SCT_cf]
    sol: OdeSolution | None
    t_events: list[npt.NDArray[np.float64]] | None
    y_events: list[npt.NDArray[_SCT_cf]] | None
    nfev: int
    njev: int
    nlu: int
    status: Literal[-1, 0, 1]
    message: str
    success: bool

def prepare_events(events: UntypedCallable | Sequence[UntypedCallable]) -> tuple[Untyped, Untyped, Untyped]: ...
def solve_event_equation(event: Untyped, sol: Untyped, t_old: Untyped, t: Untyped) -> Untyped: ...
def handle_events(
    sol: DenseOutput,
    events: list[UntypedCallable],
    active_events: Untyped,
    event_count: Untyped,
    max_events: Untyped,
    t_old: Untyped,
    t: Untyped,
) -> tuple[Untyped, Untyped, Untyped]: ...
def find_active_events(g: Untyped, g_new: Untyped, direction: Untyped) -> UntypedArray: ...
def solve_ivp(
    fun: UntypedCallable,
    t_span: Sequence[op.CanFloat],
    y0: npt.ArrayLike,
    method: Literal["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"] | type[OdeSolver] = "RK45",
    t_eval: npt.ArrayLike | None = None,
    dense_output: bool = False,
    events: UntypedCallable | list[UntypedCallable] | None = None,
    vectorized: bool = False,
    args: UntypedTuple | None = None,
    **options: Untyped,  # TODO: TypedDict in `scipy.integrate._typing`
) -> Untyped: ...
