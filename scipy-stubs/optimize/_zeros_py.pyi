from typing import Final

from scipy._typing import Untyped
from ._optimize import OptimizeResult as OptimizeResult

__all__ = ["RootResults", "bisect", "brenth", "brentq", "newton", "ridder", "toms748"]

CONVERGED: Final = "converged"
SIGNERR: Final = "sign error"
CONVERR: Final = "convergence error"
VALUEERR: Final = "value error"
INPROGRESS: Final = "No error"
flag_map: Final[dict[int, str]]

class RootResults(OptimizeResult):
    root: Untyped
    iterations: Untyped
    function_calls: Untyped
    converged: Untyped
    flag: Untyped
    method: Untyped
    def __init__(self, root, iterations, function_calls, flag, method) -> None: ...

def results_c(full_output, r, method) -> Untyped: ...
def newton(
    func,
    x0,
    fprime: Untyped | None = None,
    args=(),
    tol: float = 1.48e-08,
    maxiter: int = 50,
    fprime2: Untyped | None = None,
    x1: Untyped | None = None,
    rtol: float = 0.0,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
def bisect(f, a, b, args=(), xtol=..., rtol=..., maxiter=..., full_output: bool = False, disp: bool = True) -> Untyped: ...
def ridder(f, a, b, args=(), xtol=..., rtol=..., maxiter=..., full_output: bool = False, disp: bool = True) -> Untyped: ...
def brentq(f, a, b, args=(), xtol=..., rtol=..., maxiter=..., full_output: bool = False, disp: bool = True) -> Untyped: ...
def brenth(f, a, b, args=(), xtol=..., rtol=..., maxiter=..., full_output: bool = False, disp: bool = True) -> Untyped: ...

class TOMS748Solver:
    f: Untyped
    args: Untyped
    function_calls: int
    iterations: int
    k: int
    ab: Untyped
    fab: Untyped
    d: Untyped
    fd: Untyped
    e: Untyped
    fe: Untyped
    disp: bool
    xtol: Untyped
    rtol: Untyped
    maxiter: Untyped
    def __init__(self) -> None: ...
    def configure(self, xtol, rtol, maxiter, disp, k): ...
    def get_result(self, x, flag=...) -> Untyped: ...
    def start(self, f, a, b, args=()) -> Untyped: ...
    def get_status(self) -> Untyped: ...
    def iterate(self) -> Untyped: ...
    def solve(self, f, a, b, args=(), xtol=..., rtol=..., k: int = 2, maxiter=..., disp: bool = True) -> Untyped: ...

def toms748(
    f,
    a,
    b,
    args=(),
    k: int = 1,
    xtol=...,
    rtol=...,
    maxiter=...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
