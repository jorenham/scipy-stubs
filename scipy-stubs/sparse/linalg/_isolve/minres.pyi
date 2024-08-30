from scipy._typing import Untyped
from .utils import make_system as make_system

def minres(
    A,
    b,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    shift: float = 0.0,
    maxiter: Untyped | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    show: bool = False,
    check: bool = False,
) -> Untyped: ...
