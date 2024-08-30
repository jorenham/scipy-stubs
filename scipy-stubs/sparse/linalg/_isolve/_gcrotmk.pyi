from scipy._typing import Untyped
from scipy.linalg import (
    get_blas_funcs as get_blas_funcs,
    lstsq as lstsq,
    qr as qr,
    qr_insert as qr_insert,
    solve as solve,
    svd as svd,
)
from scipy.sparse.linalg._isolve.utils import make_system as make_system

def gcrotmk(
    A,
    b,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int = 1000,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    m: int = 20,
    k: Untyped | None = None,
    CU: Untyped | None = None,
    discard_C: bool = False,
    truncate: str = "oldest",
) -> Untyped: ...
