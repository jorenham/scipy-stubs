from scipy._typing import Untyped

__all__ = ["lsqr"]

eps: float = ...

def lsqr(
    A: Untyped,
    b: Untyped,
    damp: float = 0.0,
    atol: float = 1e-06,
    btol: float = 1e-06,
    conlim: float = ...,
    iter_lim: Untyped | None = None,
    show: bool = False,
    calc_var: bool = False,
    x0: Untyped | None = None,
) -> Untyped: ...
