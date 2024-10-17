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
    def __init__(self, root: Untyped, iterations: Untyped, function_calls: Untyped, flag: Untyped, method: Untyped) -> None: ...

def results_c(full_output: Untyped, r: Untyped, method: Untyped) -> Untyped: ...
def newton(
    func: Untyped,
    x0: Untyped,
    fprime: Untyped | None = None,
    args: Untyped = (),
    tol: float = 1.48e-08,
    maxiter: int = 50,
    fprime2: Untyped | None = None,
    x1: Untyped | None = None,
    rtol: float = 0.0,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
def bisect(
    f: Untyped,
    a: Untyped,
    b: Untyped,
    args: Untyped = (),
    xtol: Untyped = ...,
    rtol: Untyped = ...,
    maxiter: Untyped = ...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
def ridder(
    f: Untyped,
    a: Untyped,
    b: Untyped,
    args: Untyped = (),
    xtol: Untyped = ...,
    rtol: Untyped = ...,
    maxiter: Untyped = ...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
def brentq(
    f: Untyped,
    a: Untyped,
    b: Untyped,
    args: Untyped = (),
    xtol: Untyped = ...,
    rtol: Untyped = ...,
    maxiter: Untyped = ...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
def brenth(
    f: Untyped,
    a: Untyped,
    b: Untyped,
    args: Untyped = (),
    xtol: Untyped = ...,
    rtol: Untyped = ...,
    maxiter: Untyped = ...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...

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
    def configure(self, xtol: Untyped, rtol: Untyped, maxiter: Untyped, disp: Untyped, k: Untyped) -> None: ...
    def get_result(self, x: Untyped, flag: Untyped = ...) -> Untyped: ...
    def start(self, f: Untyped, a: Untyped, b: Untyped, args: Untyped = ()) -> Untyped: ...
    def get_status(self) -> Untyped: ...
    def iterate(self) -> Untyped: ...
    def solve(
        self,
        f: Untyped,
        a: Untyped,
        b: Untyped,
        args: Untyped = (),
        xtol: Untyped = ...,
        rtol: Untyped = ...,
        k: int = 2,
        maxiter: Untyped = ...,
        disp: bool = True,
    ) -> Untyped: ...

def toms748(
    f: Untyped,
    a: Untyped,
    b: Untyped,
    args: Untyped = (),
    k: int = 1,
    xtol: Untyped = ...,
    rtol: Untyped = ...,
    maxiter: Untyped = ...,
    full_output: bool = False,
    disp: bool = True,
) -> Untyped: ...
