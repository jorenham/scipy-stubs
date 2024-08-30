from scipy._typing import Untyped
from scipy.sparse.linalg._interface import aslinearoperator as aslinearoperator

def lsmr(
    A,
    b,
    damp: float = 0.0,
    atol: float = 1e-06,
    btol: float = 1e-06,
    conlim: float = ...,
    maxiter: Untyped | None = None,
    show: bool = False,
    x0: Untyped | None = None,
) -> Untyped: ...
