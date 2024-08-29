from scipy._typing import Untyped

__all__ = ["approx_jacobian", "fmin_slsqp"]

def approx_jacobian(x: Untyped, func: Untyped, epsilon: Untyped, *args: Untyped) -> Untyped: ...
def fmin_slsqp(
    func: Untyped,
    x0: Untyped,
    eqcons: Untyped = ...,
    f_eqcons: Untyped | None = None,
    ieqcons: Untyped = ...,
    f_ieqcons: Untyped | None = None,
    bounds: Untyped = ...,
    fprime: Untyped | None = None,
    fprime_eqcons: Untyped | None = None,
    fprime_ieqcons: Untyped | None = None,
    args: Untyped = ...,
    iter: int = 100,
    acc: float = 1e-06,
    iprint: int = 1,
    disp: Untyped | None = None,
    full_output: int = 0,
    epsilon: Untyped = ...,
    callback: Untyped | None = None,
) -> Untyped: ...
