from collections.abc import Callable
from typing import Final, Literal

from scipy._typing import Untyped, UntypedCallable
from scipy.optimize import OptimizeResult

TERMINATION_MESSAGES: Final[dict[Literal[-1, 0, 1, 2, 3, 4], str]]
FROM_MINPACK_TO_COMMON: Final[dict[Literal[0, 1, 2, 3, 4, 5], Literal[-1, 2, 3, 4, 1, 0]]]
IMPLEMENTED_LOSSES: Final[dict[str, Callable[[Untyped, Untyped, bool], None] | None]]

def call_minpack(
    fun: UntypedCallable,
    x0: Untyped,
    jac: Untyped,
    ftol: float,
    xtol: float,
    gtol: float,
    max_nfev: int,
    x_scale: Untyped,
    diff_step: Untyped,
) -> Untyped: ...
def prepare_bounds(bounds: Untyped, n: Untyped) -> Untyped: ...
def check_tolerance(ftol: float, xtol: float, gtol: float, method: Untyped) -> Untyped: ...
def check_x_scale(x_scale: Untyped, x0: Untyped) -> Untyped: ...
def check_jac_sparsity(jac_sparsity: Untyped, m: Untyped, n: Untyped) -> Untyped: ...
def huber(z: Untyped, rho: Untyped, cost_only: bool) -> None: ...
def soft_l1(z: Untyped, rho: Untyped, cost_only: bool) -> None: ...
def cauchy(z: Untyped, rho: Untyped, cost_only: bool) -> None: ...
def arctan(z: Untyped, rho: Untyped, cost_only: bool) -> None: ...
def construct_loss_function(m: Untyped, loss: Untyped, f_scale: Untyped) -> Untyped: ...
def least_squares(
    fun: UntypedCallable,
    x0: Untyped,
    jac: str = "2-point",
    bounds: tuple[float, float] = ...,
    method: str = "trf",
    ftol: float = 1e-08,
    xtol: float = 1e-08,
    gtol: float = 1e-08,
    x_scale: float = 1.0,
    loss: str = "linear",
    f_scale: float = 1.0,
    diff_step: Untyped | None = None,
    tr_solver: Untyped | None = None,
    tr_options: dict[str, object] = ...,
    jac_sparsity: Untyped | None = None,
    max_nfev: int | None = None,
    verbose: Literal[False, 0, True, 1] = 0,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] = ...,
) -> OptimizeResult: ...
