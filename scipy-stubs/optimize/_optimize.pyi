from scipy._lib._util import MapWrapper, _RichResult
from scipy._typing import Untyped
from ._linesearch import line_search_wolfe2 as line_search

__all__ = [
    "OptimizeResult",
    "OptimizeWarning",
    "approx_fprime",
    "bracket",
    "brent",
    "brute",
    "check_grad",
    "fmin",
    "fmin_bfgs",
    "fmin_cg",
    "fmin_ncg",
    "fmin_powell",
    "fminbound",
    "golden",
    "line_search",
    "rosen",
    "rosen_der",
    "rosen_hess",
    "rosen_hess_prod",
    "show_options",
]

class MemoizeJac:
    fun: Untyped
    jac: Untyped
    x: Untyped
    def __init__(self, fun) -> None: ...
    def __call__(self, x, *args) -> Untyped: ...
    def derivative(self, x, *args) -> Untyped: ...

class OptimizeResult(_RichResult): ...
class OptimizeWarning(UserWarning): ...

def is_finite_scalar(x) -> Untyped: ...
def vecnorm(x, ord: int = 2) -> Untyped: ...
def rosen(x) -> Untyped: ...
def rosen_der(x) -> Untyped: ...
def rosen_hess(x) -> Untyped: ...
def rosen_hess_prod(x, p) -> Untyped: ...

class _MaxFuncCallError(RuntimeError): ...

def fmin(
    func,
    x0,
    args=(),
    xtol: float = 0.0001,
    ftol: float = 0.0001,
    maxiter: Untyped | None = None,
    maxfun: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    initial_simplex: Untyped | None = None,
) -> Untyped: ...
def approx_fprime(xk, f, epsilon=..., *args) -> Untyped: ...
def check_grad(func, grad, x0, *args, epsilon=..., direction: str = "all", seed: Untyped | None = None) -> Untyped: ...
def approx_fhess_p(x0, p, fprime, epsilon, *args) -> Untyped: ...

class _LineSearchError(RuntimeError): ...

def fmin_bfgs(
    f,
    x0,
    fprime: Untyped | None = None,
    args=(),
    gtol: float = 1e-05,
    norm=...,
    epsilon=...,
    maxiter: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    xrtol: int = 0,
    c1: float = 0.0001,
    c2: float = 0.9,
    hess_inv0: Untyped | None = None,
) -> Untyped: ...
def fmin_cg(
    f,
    x0,
    fprime: Untyped | None = None,
    args=(),
    gtol: float = 1e-05,
    norm=...,
    epsilon=...,
    maxiter: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.4,
) -> Untyped: ...
def fmin_ncg(
    f,
    x0,
    fprime,
    fhess_p: Untyped | None = None,
    fhess: Untyped | None = None,
    args=(),
    avextol: float = 1e-05,
    epsilon=...,
    maxiter: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.9,
) -> Untyped: ...
def fminbound(func, x1, x2, args=(), xtol: float = 1e-05, maxfun: int = 500, full_output: int = 0, disp: int = 1) -> Untyped: ...

class Brent:
    func: Untyped
    args: Untyped
    tol: Untyped
    maxiter: Untyped
    xmin: Untyped
    fval: Untyped
    iter: int
    funcalls: int
    disp: Untyped
    def __init__(self, func, args=(), tol: float = 1.48e-08, maxiter: int = 500, full_output: int = 0, disp: int = 0): ...
    brack: Untyped
    def set_bracket(self, brack: Untyped | None = None): ...
    def get_bracket_info(self) -> Untyped: ...
    def optimize(self): ...
    def get_result(self, full_output: bool = False) -> Untyped: ...

def brent(
    func, args=(), brack: Untyped | None = None, tol: float = 1.48e-08, full_output: int = 0, maxiter: int = 500
) -> Untyped: ...
def golden(func, args=(), brack: Untyped | None = None, tol=..., full_output: int = 0, maxiter: int = 5000) -> Untyped: ...
def bracket(func, xa: float = 0.0, xb: float = 1.0, args=(), grow_limit: float = 110.0, maxiter: int = 1000) -> Untyped: ...

class BracketError(RuntimeError): ...

def fmin_powell(
    func,
    x0,
    args=(),
    xtol: float = 0.0001,
    ftol: float = 0.0001,
    maxiter: Untyped | None = None,
    maxfun: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    direc: Untyped | None = None,
) -> Untyped: ...
def brute(
    func, ranges, args=(), Ns: int = 20, full_output: int = 0, finish=..., disp: bool = False, workers: int = 1
) -> Untyped: ...

class _Brute_Wrapper:
    f: Untyped
    args: Untyped
    def __init__(self, f, args) -> None: ...
    def __call__(self, x) -> Untyped: ...

def show_options(solver: Untyped | None = None, method: Untyped | None = None, disp: bool = True) -> Untyped: ...
