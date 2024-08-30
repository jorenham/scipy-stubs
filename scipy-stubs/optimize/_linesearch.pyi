from scipy._typing import Untyped
from ._dcsrch import DCSRCH as DCSRCH

class LineSearchWarning(RuntimeWarning): ...

def line_search_wolfe1(
    f,
    fprime,
    xk,
    pk,
    gfk: Untyped | None = None,
    old_fval: Untyped | None = None,
    old_old_fval: Untyped | None = None,
    args=(),
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: int = 50,
    amin: float = 1e-08,
    xtol: float = 1e-14,
) -> Untyped: ...
def scalar_search_wolfe1(
    phi,
    derphi,
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
    f,
    myfprime,
    xk,
    pk,
    gfk: Untyped | None = None,
    old_fval: Untyped | None = None,
    old_old_fval: Untyped | None = None,
    args=(),
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: Untyped | None = None,
    extra_condition: Untyped | None = None,
    maxiter: int = 10,
) -> Untyped: ...
def scalar_search_wolfe2(
    phi,
    derphi,
    phi0: Untyped | None = None,
    old_phi0: Untyped | None = None,
    derphi0: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: Untyped | None = None,
    extra_condition: Untyped | None = None,
    maxiter: int = 10,
) -> Untyped: ...
def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1: float = 0.0001, alpha0: int = 1) -> Untyped: ...
def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1: float = 0.0001, alpha0: int = 1) -> Untyped: ...
def scalar_search_armijo(phi, phi0, derphi0, c1: float = 0.0001, alpha0: int = 1, amin: int = 0) -> Untyped: ...
