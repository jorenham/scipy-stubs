from scipy._typing import Untyped

__all__ = ["curve_fit", "fixed_point", "fsolve", "leastsq"]

def fsolve(
    func,
    x0,
    args=(),
    fprime: Untyped | None = None,
    full_output: int = 0,
    col_deriv: int = 0,
    xtol: float = ...,
    maxfev: int = 0,
    band: Untyped | None = None,
    epsfcn: Untyped | None = None,
    factor: int = 100,
    diag: Untyped | None = None,
) -> Untyped: ...
def leastsq(
    func,
    x0,
    args=(),
    Dfun: Untyped | None = None,
    full_output: bool = False,
    col_deriv: bool = False,
    ftol: float = ...,
    xtol: float = ...,
    gtol: float = 0.0,
    maxfev: int = 0,
    epsfcn: Untyped | None = None,
    factor: int = 100,
    diag: Untyped | None = None,
) -> Untyped: ...
def curve_fit(
    f,
    xdata,
    ydata,
    p0: Untyped | None = None,
    sigma: Untyped | None = None,
    absolute_sigma: bool = False,
    check_finite: Untyped | None = None,
    bounds=...,
    method: Untyped | None = None,
    jac: Untyped | None = None,
    *,
    full_output: bool = False,
    nan_policy: Untyped | None = None,
    **kwargs,
) -> Untyped: ...
def check_gradient(fcn, Dfcn, x0, args=(), col_deriv: int = 0) -> Untyped: ...
def fixed_point(func, x0, args=(), xtol: float = 1e-08, maxiter: int = 500, method: str = "del2") -> Untyped: ...
