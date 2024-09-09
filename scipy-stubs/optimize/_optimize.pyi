from typing import Any, Final, Literal

import numpy as np
import optype.numpy as onpt
from scipy._lib._util import _RichResult
from scipy._typing import Untyped, UntypedCallable, UntypedTuple
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

class _LineSearchError(RuntimeError): ...
class _MaxFuncCallError(RuntimeError): ...
class BracketError(RuntimeError): ...
class OptimizeWarning(UserWarning): ...

class _Brute_Wrapper:
    f: UntypedCallable
    args: Untyped
    def __init__(self, f: UntypedCallable, args: Untyped) -> None: ...
    def __call__(self, x: Untyped) -> Untyped: ...

class MemoizeJac:
    fun: UntypedCallable
    jac: Untyped
    x: Untyped
    def __init__(self, fun: UntypedCallable) -> None: ...
    def __call__(self, x: Untyped, *args: Untyped) -> Untyped: ...
    def derivative(self, x: Untyped, *args: Untyped) -> Untyped: ...

class Brent:
    func: UntypedCallable
    args: Untyped
    tol: float
    maxiter: int
    xmin: Untyped
    fval: Untyped
    iter: int
    funcalls: int
    disp: bool | Literal[0, 1]
    brack: Untyped
    def __init__(
        self,
        func: UntypedCallable,
        args: Untyped = (),
        tol: float = 1.48e-08,
        maxiter: int = 500,
        full_output: int = 0,
        disp: bool | Literal[0, 1] = 0,
    ) -> None: ...
    def set_bracket(self, brack: Untyped | None = None) -> None: ...
    def get_bracket_info(self) -> Untyped: ...
    def optimize(self) -> None: ...
    def get_result(self, full_output: bool = False) -> Untyped: ...

class OptimizeResult(_RichResult):
    x: Final[onpt.Array[tuple[int], np.floating[Any]]]
    success: Final[bool]
    status: int
    message: str
    fun: float | np.floating[Any]
    jac: onpt.Array[tuple[int, ...], np.floating[Any]]
    hess: onpt.Array[tuple[int, ...], np.floating[Any]]
    nfev: int
    njev: int
    nhev: int
    nit: int
    maxcv: float

def is_finite_scalar(x: Untyped) -> Untyped: ...
def vecnorm(x: Untyped, ord: int = 2) -> Untyped: ...
def rosen(x: Untyped) -> Untyped: ...
def rosen_der(x: Untyped) -> Untyped: ...
def rosen_hess(x: Untyped) -> Untyped: ...
def rosen_hess_prod(x: Untyped, p: Untyped) -> Untyped: ...
def fmin(
    func: UntypedCallable,
    x0: Untyped,
    args: Untyped = (),
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
def approx_fprime(xk: Untyped, f: UntypedCallable, epsilon: float = ..., *args: Untyped) -> Untyped: ...
def check_grad(
    func: UntypedCallable,
    grad: Untyped,
    x0: Untyped,
    *args: Untyped,
    epsilon: float = ...,
    direction: str = "all",
    seed: Untyped | None = None,
) -> Untyped: ...
def approx_fhess_p(x0: Untyped, p: Untyped, fprime: Untyped, epsilon: float, *args: Untyped) -> Untyped: ...
def fmin_bfgs(
    f: UntypedCallable,
    x0: Untyped,
    fprime: Untyped | None = None,
    args: Untyped = (),
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: float = ...,
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
    f: UntypedCallable,
    x0: Untyped,
    fprime: Untyped | None = None,
    args: Untyped = (),
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: float = ...,
    maxiter: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.4,
) -> Untyped: ...
def fmin_ncg(
    f: UntypedCallable,
    x0: Untyped,
    fprime: Untyped,
    fhess_p: Untyped | None = None,
    fhess: Untyped | None = None,
    args: Untyped = (),
    avextol: float = 1e-05,
    epsilon: float = ...,
    maxiter: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: Untyped | None = None,
    c1: float = 0.0001,
    c2: float = 0.9,
) -> Untyped: ...
def fminbound(
    func: UntypedCallable,
    x1: Untyped,
    x2: Untyped,
    args: Untyped = (),
    xtol: float = 1e-05,
    maxfun: int = 500,
    full_output: int = 0,
    disp: int = 1,
) -> Untyped: ...
def brent(
    func: UntypedCallable,
    args: Untyped = (),
    brack: Untyped | None = None,
    tol: float = 1.48e-08,
    full_output: int = 0,
    maxiter: int = 500,
) -> Untyped: ...
def golden(
    func: UntypedCallable,
    args: Untyped = (),
    brack: Untyped | None = None,
    tol: float = ...,
    full_output: int = 0,
    maxiter: int = 5000,
) -> Untyped: ...
def bracket(
    func: UntypedCallable,
    xa: float = 0.0,
    xb: float = 1.0,
    args: Untyped = (),
    grow_limit: float = 110.0,
    maxiter: int = 1000,
) -> Untyped: ...
def fmin_powell(
    func: UntypedCallable,
    x0: Untyped,
    args: Untyped = (),
    xtol: float = 0.0001,
    ftol: float = 0.0001,
    maxiter: Untyped | None = None,
    maxfun: Untyped | None = None,
    full_output: int = 0,
    disp: int = 1,
    retall: int = 0,
    callback: UntypedCallable | None = None,
    direc: Untyped | None = None,
) -> Untyped: ...
def brute(
    func: UntypedCallable,
    ranges: Untyped,
    args: UntypedTuple = (),
    Ns: int = 20,
    full_output: int = 0,
    finish: Untyped = ...,
    disp: bool = False,
    workers: int = 1,
) -> Untyped: ...
def show_options(solver: str | None = None, method: str | None = None, disp: bool = True) -> str | None: ...
