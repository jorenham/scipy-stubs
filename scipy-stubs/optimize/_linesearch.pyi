from scipy._typing import Untyped, UntypedCallable

__all__ = [
    "LineSearchWarning",
    "line_search_armijo",
    "line_search_wolfe1",
    "line_search_wolfe2",
    "scalar_search_wolfe1",
    "scalar_search_wolfe2",
]

class LineSearchWarning(RuntimeWarning): ...

def line_search_wolfe1(
    f: UntypedCallable,
    fprime: Untyped,
    xk: Untyped,
    pk: Untyped,
    gfk: Untyped | None = None,
    old_fval: Untyped | None = None,
    old_old_fval: Untyped | None = None,
    args: Untyped = (),
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: int = 50,
    amin: float = 1e-08,
    xtol: float = 1e-14,
) -> Untyped: ...
def scalar_search_wolfe1(
    phi: Untyped,
    derphi: Untyped,
    phi0: Untyped | None = None,
    old_phi0: Untyped | None = None,
    derphi0: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: int = 50,
    amin: float = 1e-08,
    xtol: float = 1e-14,
) -> Untyped: ...

line_search = line_search_wolfe1

def line_search_wolfe2(
    f: UntypedCallable,
    myfprime: Untyped,
    xk: Untyped,
    pk: Untyped,
    gfk: Untyped | None = None,
    old_fval: Untyped | None = None,
    old_old_fval: Untyped | None = None,
    args: Untyped = (),
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: Untyped | None = None,
    extra_condition: Untyped | None = None,
    maxiter: int = 10,
) -> Untyped: ...
def scalar_search_wolfe2(
    phi: Untyped,
    derphi: Untyped,
    phi0: Untyped | None = None,
    old_phi0: Untyped | None = None,
    derphi0: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: Untyped | None = None,
    extra_condition: Untyped | None = None,
    maxiter: int = 10,
) -> Untyped: ...
def line_search_armijo(
    f: UntypedCallable,
    xk: Untyped,
    pk: Untyped,
    gfk: Untyped,
    old_fval: Untyped,
    args: Untyped = (),
    c1: float = 0.0001,
    alpha0: int = 1,
) -> Untyped: ...
def line_search_BFGS(
    f: UntypedCallable,
    xk: Untyped,
    pk: Untyped,
    gfk: Untyped,
    old_fval: Untyped,
    args: Untyped = (),
    c1: float = 0.0001,
    alpha0: int = 1,
) -> Untyped: ...
def scalar_search_armijo(
    phi: Untyped,
    phi0: Untyped,
    derphi0: Untyped,
    c1: float = 0.0001,
    alpha0: int = 1,
    amin: int = 0,
) -> Untyped: ...
