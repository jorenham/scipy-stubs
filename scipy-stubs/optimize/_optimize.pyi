from collections.abc import Callable
from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar, TypeVarTuple, Unpack

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
import scipy._typing as spt
from scipy._lib._util import _RichResult
from scipy._typing import Untyped, UntypedCallable
from ._linesearch import line_search_wolfe2 as line_search
from ._typing import Brack, MethodAll, ObjectiveFunc, Solver

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

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[object, ...]])
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any], default=np.float32 | np.float64)

_Float_co: TypeAlias = float | np.floating[Any] | np.integer[Any]
_FloatND_co: TypeAlias = onpt.Array[tuple[int, ...], np.floating[Any] | np.integer[Any]]

_Float: TypeAlias = float | np.float64
_Float0D: TypeAlias = _Float | onpt.Array[tuple[()], np.float64]
_Float1D: TypeAlias = onpt.Array[tuple[int], np.float64]
_Float2D: TypeAlias = onpt.Array[tuple[int, int], np.float64]

# undocumented

def is_finite_scalar(x: complex | np.generic) -> bool | np.bool_: ...  # undocumented
def vecnorm(x: _ArrayLikeFloat_co, ord: _Float_co | np.integer[Any] = 2) -> _Float_co: ...  # undocumented
def approx_fhess_p(  # undocumented
    x0: _ArrayLikeFloat_co,
    p: _Float_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co],
    epsilon: _Float_co | _FloatND_co,
    *args: Unpack[_Ts],
) -> _Float1D: ...

class MemoizeJac(Generic[Unpack[_Ts]]):  # undocumented
    fun: ObjectiveFunc[_ArrayLikeFloat_co, Unpack[_Ts], tuple[_Float_co, _FloatND_co]]
    jac: float | None
    x: onpt.Array[onpt.AtMost1D, np.floating[Any]] | None
    _value: float
    def __init__(self, /, fun: ObjectiveFunc[_ArrayLikeFloat_co, Unpack[_Ts], tuple[_Float_co, _FloatND_co]]) -> None: ...
    def _compute_if_needed(self, /, x: _ArrayLikeFloat_co, *args: Unpack[_Ts]) -> None: ...
    def __call__(self, /, x: _ArrayLikeFloat_co, *args: Unpack[_Ts]) -> _Float0D: ...
    def derivative(self, /, x: _ArrayLikeFloat_co, *args: Unpack[_Ts]) -> _Float1D: ...

class Brent(Generic[Unpack[_Ts]]):  # undocumented
    _mintol: float
    _cg: float
    func: ObjectiveFunc[float, Unpack[_Ts], float]
    args: tuple[Unpack[_Ts]]
    tol: float
    maxiter: int
    xmin: float | None
    fval: float | None
    iter: int
    funcalls: int
    disp: spt.AnyBool
    brack: tuple[float, float] | tuple[float, float, float]
    @overload
    def __init__(
        self: Brent[Unpack[tuple[()]]],
        /,
        func: ObjectiveFunc[float, Unpack[tuple[()]], float],
        args: tuple[()] = (),
        tol: float = 1.48e-08,
        maxiter: int = 500,
        full_output: spt.AnyBool = 0,
        disp: spt.AnyBool = 0,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        func: ObjectiveFunc[float, Unpack[_Ts], float],
        args: tuple[Unpack[_Ts]] = ...,
        tol: float = 1.48e-08,
        maxiter: int = 500,
        full_output: spt.AnyBool = 0,
        disp: spt.AnyBool = 0,
    ) -> None: ...
    def set_bracket(self, /, brack: tuple[float, float] | tuple[float, float, float] | None = None) -> None: ...
    def get_bracket_info(self, /) -> tuple[float, float, float, float, float, float, int]: ...
    def optimize(self, /) -> None: ...
    def get_result(self, /, full_output: bool = False) -> float | tuple[float, float, int, int]: ...

# minimize

@overload
def fmin(
    func: ObjectiveFunc[_Float1D, Unpack[tuple[()]], _Float_co],
    x0: _ArrayLikeFloat_co,
    args: tuple[()] = (),
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> _Float1D | tuple[_Float1D, _Float_co] | tuple[_Float1D, _Float_co, int, int, int, list[_Float1D]]: ...
@overload
def fmin(
    func: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    x0: _ArrayLikeFloat_co,
    args: tuple[Unpack[_Ts]],
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> _Float1D | tuple[_Float1D, _Float_co] | tuple[_Float1D, _Float_co, int, int, int, list[_Float1D]]: ...
@overload
def fmin_bfgs(
    f: ObjectiveFunc[_Float1D, Unpack[tuple[()]], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[()] = (),
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    xrtol: float = 0,
    c1: float = 1e-4,
    c2: float = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, _Float1D, _Float2D, int, int, int]
    | tuple[_Float1D, _Float0D, _Float1D, _Float2D, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_bfgs(
    f: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[Unpack[_Ts]] = ...,
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    xrtol: float = 0,
    c1: float = 1e-4,
    c2: float = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, _Float1D, _Float2D, int, int, int]
    | tuple[_Float1D, _Float0D, _Float1D, _Float2D, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_cg(
    f: ObjectiveFunc[_Float1D, Unpack[tuple[()]], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[()] = (),
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    c1: float = 1e-4,
    c2: float = 0.4,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, int, int, int]
    | tuple[_Float1D, _Float0D, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_cg(
    f: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[Unpack[_Ts]] = ...,
    gtol: float = 1e-05,
    norm: float = ...,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    c1: float = 1e-4,
    c2: float = 0.4,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, int, int, int]
    | tuple[_Float1D, _Float0D, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_ncg(
    f: ObjectiveFunc[_Float1D, Unpack[tuple[()]], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co],
    fhess_p: ObjectiveFunc[_Float1D, _Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    fhess: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[()] = (),
    avextol: float = 1e-05,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, int, int, int, int]
    | tuple[_Float1D, _Float0D, int, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_ncg(
    f: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    x0: _ArrayLikeFloat_co,
    fprime: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co],
    fhess_p: ObjectiveFunc[_Float1D, _Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    fhess: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co] | None = None,
    args: tuple[Unpack[_Ts]] = ...,
    avextol: float = 1e-05,
    epsilon: _Float_co | _FloatND_co = ...,
    maxiter: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, int, int, int, int]
    | tuple[_Float1D, _Float0D, int, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_powell(
    func: ObjectiveFunc[_Float1D, Unpack[tuple[()]], _Float_co],
    x0: _ArrayLikeFloat_co,
    args: tuple[()] = (),
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, _Float2D, int, int, int]
    | tuple[_Float1D, _Float0D, _Float2D, int, int, int, list[_Float1D]]
): ...
@overload
def fmin_powell(
    func: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    x0: _ArrayLikeFloat_co,
    args: tuple[Unpack[_Ts]] = ...,
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: spt.AnyBool = 0,
    disp: spt.AnyBool = 1,
    retall: spt.AnyBool = 0,
    callback: Callable[[_Float1D], None] | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> (
    _Float1D
    | tuple[_Float1D, _Float0D]
    | tuple[_Float1D, _Float0D, _Float2D, int, int, int]
    | tuple[_Float1D, _Float0D, _Float2D, int, int, int, list[_Float1D]]
): ...
@overload
def fminbound(
    func: ObjectiveFunc[float, Unpack[tuple[()]], _Float_co],
    x1: _Float_co,
    x2: _Float_co,
    args: tuple[()] = (),
    xtol: float = 1e-05,
    maxfun: int = 500,
    full_output: spt.AnyBool = 0,
    disp: Literal[0, 1, 2, 3] = 1,
) -> float | tuple[float, _Float0D, Literal[0, 1], int]: ...
@overload
def fminbound(
    func: ObjectiveFunc[float, Unpack[_Ts], _Float_co],
    x1: _Float_co,
    x2: _Float_co,
    args: tuple[Unpack[_Ts]] = ...,
    xtol: float = 1e-05,
    maxfun: int = 500,
    full_output: spt.AnyBool = 0,
    disp: Literal[0, 1, 2, 3] = 1,
) -> float | tuple[float, _Float0D, Literal[0, 1], int]: ...

# global minimization: TODO

def brute(
    func: UntypedCallable,
    ranges: Untyped,
    args: Untyped = (),
    Ns: int = 20,
    full_output: int = 0,
    finish: Untyped = ...,
    disp: bool = False,
    workers: int = 1,
) -> Untyped: ...

# minimize scalar

@overload
def brent(
    func: ObjectiveFunc[float, Unpack[tuple[()]], _Float_co],
    args: tuple[()] = (),
    brack: Brack | None = None,
    tol: float = 1.48e-08,
    full_output: spt.AnyBool = 0,
    maxiter: int = 500,
) -> float | tuple[float, _Float0D, int, int]: ...
@overload
def brent(
    func: ObjectiveFunc[float, Unpack[_Ts], _Float_co],
    args: tuple[Unpack[_Ts]],
    brack: Brack | None = None,
    tol: float = 1.48e-08,
    full_output: spt.AnyBool = 0,
    maxiter: int = 500,
) -> float | tuple[float, _Float0D, int, int]: ...
@overload
def golden(
    func: ObjectiveFunc[float, Unpack[tuple[()]], _Float_co],
    args: tuple[()] = (),
    brack: Brack | None = None,
    tol: float = ...,
    full_output: spt.AnyBool = 0,
    maxiter: int = 5_000,
) -> float | tuple[float, _Float0D, int]: ...
@overload
def golden(
    func: ObjectiveFunc[float, Unpack[_Ts], _Float_co],
    args: tuple[Unpack[_Ts]],
    brack: Brack | None = None,
    tol: float = ...,
    full_output: spt.AnyBool = 0,
    maxiter: int = 5_000,
) -> float | tuple[float, _Float0D, int]: ...
@overload
def bracket(
    func: ObjectiveFunc[float, Unpack[tuple[()]], _Float_co],
    xa: _Float_co = 0.0,
    xb: _Float_co = 1.0,
    args: tuple[()] = (),
    grow_limit: float = 110.0,
    maxiter: int = 1_000,
) -> tuple[float, float, float, _Float0D, _Float0D, _Float0D, int]: ...
@overload
def bracket(
    func: ObjectiveFunc[float, Unpack[_Ts], _Float_co],
    xa: _Float_co = 0.0,
    xb: _Float_co = 1.0,
    args: tuple[Unpack[_Ts]] = ...,
    grow_limit: float = 110.0,
    maxiter: int = 1_000,
) -> tuple[float, float, float, _Float0D, _Float0D, _Float0D, int]: ...

# rosenbrock function

def rosen(x: _ArrayLikeFloat_co) -> _Float_co: ...
def rosen_der(x: _ArrayLikeFloat_co) -> _Float1D: ...
def rosen_hess(x: _ArrayLikeFloat_co) -> _Float2D: ...
def rosen_hess_prod(x: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co) -> _Float1D: ...

# meta

@overload
def show_options(solver: Solver | None, method: MethodAll | None, disp: Literal[False, 0, None]) -> str: ...
@overload
def show_options(solver: Solver | None = None, method: MethodAll | None = None, *, disp: Literal[False, 0, None]) -> str: ...
@overload
def show_options(solver: Solver | None = None, method: MethodAll | None = None, disp: Literal[True, 1] = True) -> None: ...

class BracketError(RuntimeError): ...
class OptimizeWarning(UserWarning): ...

_NT = TypeVar("_NT", bound=int, default=int)

class OptimizeResult(_RichResult, Generic[_NT, _SCT_f]):
    x: onpt.Array[tuple[_NT], _SCT_f]
    fun: float | _SCT_f
    success: bool
    status: int
    message: str
    nit: int

# miscellaneous

def approx_fprime(
    xk: _ArrayLikeFloat_co,
    f: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    epsilon: _ArrayLikeFloat_co = ...,
    *args: Unpack[_Ts],
) -> _Float1D: ...
def check_grad(
    func: ObjectiveFunc[_Float1D, Unpack[_Ts], _Float_co],
    grad: ObjectiveFunc[_Float1D, Unpack[_Ts], _FloatND_co],
    x0: _ArrayLikeFloat_co,
    *args: tuple[Unpack[_Ts]],
    epsilon: float = ...,
    direction: Literal["all", "random"] = "all",
    seed: spt.Seed | None = None,
) -> _Float: ...
