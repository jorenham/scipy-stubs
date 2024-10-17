from scipy._typing import Untyped

__all__ = ["lgmres"]

def lgmres(
    A: Untyped,
    b: Untyped,
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
