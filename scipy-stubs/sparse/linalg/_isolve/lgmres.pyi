from scipy._typing import Untyped
from scipy.linalg import get_blas_funcs as get_blas_funcs
from .utils import make_system as make_system

def lgmres(
    A,
    b,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int = 1000,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    inner_m: int = 30,
    outer_k: int = 3,
    outer_v: Untyped | None = None,
    store_outer_Av: bool = True,
    prepend_outer_v: bool = False,
) -> Untyped: ...
