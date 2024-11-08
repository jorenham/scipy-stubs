from collections.abc import Callable
from typing import Any, Concatenate, Final, Generic, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeNumber_co
from scipy._typing import AnyReal, Untyped, UntypedCallable
from ._optimize import OptimizeResult

__all__ = ["RootResults", "bisect", "brenth", "brentq", "newton", "ridder", "toms748"]

_Flag: TypeAlias = Literal["converged", "sign error", "convergence error", "value error", "No error"]

_Float: TypeAlias = float | np.float64
_Floating: TypeAlias = float | np.floating[Any]

_AnyRoot: TypeAlias = complex | np.inexact[Any] | npt.NDArray[np.inexact[Any]]
_RootT_co = TypeVar("_RootT_co", covariant=True, bound=_AnyRoot, default=_Float)

_Fn_f_0d: TypeAlias = Callable[Concatenate[float, ...], _Float] | Callable[Concatenate[np.float64, ...], _Floating]

###

CONVERGED: Final = "converged"
SIGNERR: Final = "sign error"
CONVERR: Final = "convergence error"
VALUEERR: Final = "value error"
INPROGRESS: Final = "No error"

flag_map: Final[dict[int, str]] = ...

class RootResults(OptimizeResult, Generic[_RootT_co]):
    root: _RootT_co  # readonly
    iterations: Final[int]
    function_calls: Final[int]
    converged: Final[bool]
    flag: Final[_Flag]
    method: Final[str]

    def __init__(
        self,
        /,
        root: _RootT_co,
        iterations: int,
        function_calls: int,
        flag: Literal[0, -1, -2, -3, -4, 1],
        method: str,
    ) -> None: ...

# undocumented
class TOMS748Solver:
    f: UntypedCallable
    args: tuple[object, ...]
    function_calls: int
    iterations: int
    k: int
    ab: Untyped
    fab: Untyped
    d: Untyped
    fd: Untyped
    e: Untyped
    fe: Untyped
    disp: op.CanBool
    xtol: AnyReal
    rtol: _Floating
    maxiter: op.CanIndex

    def __init__(self) -> None: ...
    def configure(self, xtol: AnyReal, rtol: _Floating, maxiter: op.CanIndex, disp: Untyped, k: Untyped) -> None: ...
    def get_result(self, x: Untyped, flag: Untyped = ...) -> Untyped: ...
    def start(self, f: UntypedCallable, a: Untyped, b: Untyped, args: tuple[object, ...] = ()) -> Untyped: ...
    def get_status(self) -> Untyped: ...
    def iterate(self) -> Untyped: ...
    def solve(
        self,
        f: UntypedCallable,
        a: Untyped,
        b: Untyped,
        args: tuple[object, ...] = (),
        xtol: AnyReal = ...,
        rtol: _Floating = ...,
        k: int = 2,
        maxiter: op.CanIndex = ...,
        disp: op.CanBool = True,
    ) -> Untyped: ...

# undocumented
def results_c(
    full_output: op.CanBool,
    r: tuple[_AnyRoot, int, int, int],
    method: str,
) -> tuple[_AnyRoot, int, int, int] | tuple[_AnyRoot, RootResults[_AnyRoot]]: ...

# TODO: overload `shape(x0)`: `() | (1) | (1, 1), ... -> root: scalar[_]`, `_ -> root: array[_]`
# TODO: overload `dtype(x0)`: `floating -> root: _[float64]`; `complexfloating -> root: _[complex128]`
# TODO: overload `full_output`: `falsy -> root`, `truthy -> (root, r, converged, zero_der)`
def newton(
    func: UntypedCallable,
    x0: _ArrayLikeNumber_co,
    fprime: UntypedCallable | None = None,
    args: tuple[object, ...] = (),
    tol: _Floating = 1.48e-08,
    maxiter: op.CanIndex = 50,
    fprime2: UntypedCallable | None = None,
    x1: _ArrayLikeNumber_co | None = None,
    rtol: _Floating = 0.0,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]: ...

# TODO: overload `full_output`: falsy | truthy => root | (root, r)
def bisect(
    f: _Fn_f_0d,
    a: AnyReal,
    b: AnyReal,
    args: tuple[object, ...] = (),
    xtol: AnyReal = 2e-12,
    rtol: _Floating = ...,
    maxiter: op.CanIndex = 100,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults]: ...
def ridder(
    f: _Fn_f_0d,
    a: AnyReal,
    b: AnyReal,
    args: tuple[object, ...] = (),
    xtol: AnyReal = 2e-12,
    rtol: _Floating = ...,
    maxiter: op.CanIndex = 100,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults]: ...
def brentq(
    f: _Fn_f_0d,
    a: AnyReal,
    b: AnyReal,
    args: tuple[object, ...] = (),
    xtol: AnyReal = 2e-12,
    rtol: _Floating = ...,
    maxiter: op.CanIndex = 100,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults]: ...
def brenth(
    f: _Fn_f_0d,
    a: AnyReal,
    b: AnyReal,
    args: tuple[object, ...] = (),
    xtol: AnyReal = 2e-12,
    rtol: _Floating = ...,
    maxiter: op.CanIndex = 100,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults]: ...
def toms748(
    f: _Fn_f_0d,
    a: AnyReal,
    b: AnyReal,
    args: tuple[object, ...] = (),
    k: int = 1,
    xtol: AnyReal = 2e-12,
    rtol: _Floating = ...,
    maxiter: op.CanIndex = 100,
    full_output: op.CanBool = False,
    disp: op.CanBool = True,
) -> _Float | tuple[_Float, RootResults]: ...
