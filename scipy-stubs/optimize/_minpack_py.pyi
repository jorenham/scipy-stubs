from scipy._typing import NanPolicy, Untyped, UntypedCallable

__all__ = ["curve_fit", "fixed_point", "fsolve", "leastsq"]

def fsolve(
    func: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    fprime: UntypedCallable | None = None,
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
    func: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    Dfun: UntypedCallable | None = None,
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
    f: UntypedCallable,
    xdata: Untyped,
    ydata: Untyped,
    p0: Untyped | None = None,
    sigma: Untyped | None = None,
    absolute_sigma: bool = False,
    check_finite: Untyped | None = None,
    bounds: Untyped = ...,
    method: Untyped | None = None,
    jac: Untyped | None = None,
    *,
    full_output: bool = False,
    nan_policy: NanPolicy | None = None,
    **kwargs: tuple[object, ...],
) -> Untyped: ...
def check_gradient(
    fcn: UntypedCallable,
    Dfcn: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    col_deriv: int = 0,
) -> Untyped: ...
def fixed_point(
    func: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    xtol: float = 1e-08,
    maxiter: int = 500,
    method: str = "del2",
) -> Untyped: ...
