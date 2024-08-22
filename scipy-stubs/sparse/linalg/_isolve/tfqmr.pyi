from .utils import make_system as make_system
from scipy._typing import Untyped

def tfqmr(
    A,
    b,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: Untyped | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    show: bool = False,
) -> Untyped: ...
