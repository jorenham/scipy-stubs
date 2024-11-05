from scipy._typing import Untyped
from scipy.sparse.linalg import LinearOperator

__all__ = ["LbfgsInvHessProduct", "fmin_l_bfgs_b"]

def fmin_l_bfgs_b(
    func: Untyped,
    x0: Untyped,
    fprime: Untyped | None = None,
    args: Untyped = (),
    approx_grad: int = 0,
    bounds: Untyped | None = None,
    m: int = 10,
    factr: float = 10000000.0,
    pgtol: float = 1e-05,
    epsilon: float = 1e-08,
    iprint: int = -1,
    maxfun: int = 15000,
    maxiter: int = 15000,
    disp: Untyped | None = None,
    callback: Untyped | None = None,
    maxls: int = 20,
) -> Untyped: ...

class LbfgsInvHessProduct(LinearOperator):
    sk: Untyped
    yk: Untyped
    n_corrs: Untyped
    rho: Untyped
    def __init__(self, sk: Untyped, yk: Untyped) -> None: ...
    def todense(self) -> Untyped: ...
