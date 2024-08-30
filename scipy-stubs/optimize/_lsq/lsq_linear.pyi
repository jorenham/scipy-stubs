from typing import Final

from scipy._typing import Untyped
from scipy.optimize import OptimizeResult

TERMINATION_MESSAGES: Final[dict[int, str]]

def prepare_bounds(bounds, n) -> Untyped: ...
def lsq_linear(
    A,
    b,
    bounds=...,
    method: str = "trf",
    tol: float = 1e-10,
    lsq_solver: Untyped | None = None,
    lsmr_tol: Untyped | None = None,
    max_iter: Untyped | None = None,
    verbose: int = 0,
    *,
    lsmr_maxiter: Untyped | None = None,
) -> OptimizeResult: ...
