from .base import OdeSolver as OdeSolver
from .bdf import BDF as BDF
from .common import EPS as EPS, OdeSolution as OdeSolution
from .lsoda import LSODA as LSODA
from .radau import Radau as Radau
from .rk import DOP853 as DOP853, RK23 as RK23, RK45 as RK45
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as OptimizeResult

METHODS: Untyped
MESSAGES: Untyped

class OdeResult(OptimizeResult): ...

def prepare_events(events) -> Untyped: ...
def solve_event_equation(event, sol, t_old, t) -> Untyped: ...
def handle_events(sol, events, active_events, event_count, max_events, t_old, t) -> Untyped: ...
def find_active_events(g, g_new, direction) -> Untyped: ...
def solve_ivp(
    fun,
    t_span,
    y0,
    method: str = "RK45",
    t_eval: Untyped | None = None,
    dense_output: bool = False,
    events: Untyped | None = None,
    vectorized: bool = False,
    args: Untyped | None = None,
    **options,
) -> Untyped: ...
