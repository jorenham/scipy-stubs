from collections.abc import Callable, Iterable, Sequence
from typing import Any, Concatenate, Final, Generic, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._lib._util import _RichResult
from scipy._typing import AnyBool, AnyInt, AnyReal, Seed
from ._linesearch import line_search_wolfe2 as line_search
from ._typing import Brack, MethodAll, Solver

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

_XT = TypeVar("_XT")
_PT = TypeVar("_PT")
_YT = TypeVar("_YT", default=AnyReal)
_VT = TypeVar("_VT")
_RT = TypeVar("_RT")

_Fn1: TypeAlias = Callable[Concatenate[_XT, ...], _YT]
_Fn1_0d: TypeAlias = _Fn1[float, _YT] | _Fn1[np.float64, _YT]
_Fn1_1d: TypeAlias = _Fn1[_Array_1d_f8, _YT]
_Fn2: TypeAlias = Callable[Concatenate[_XT, _PT, ...], _YT]
_Callback_1d: TypeAlias = Callable[[_Array_1d_f8], None]

_Falsy: TypeAlias = Literal[0, False]
_Truthy: TypeAlias = Literal[1, True]

_Scalar: TypeAlias = complex | np.number[Any] | np.bool_
_Scalar_f: TypeAlias = float | np.floating[Any]
_Scalar_f8: TypeAlias = float | np.float64  # equivalent to `np.float64` in `numpy>=2.2`

_Array: TypeAlias = onpt.Array[tuple[int, ...], np.number[Any] | np.bool_ | np.object_]
_Array_f: TypeAlias = onpt.Array[tuple[int, ...], np.floating[Any]]
_Array_f_co: TypeAlias = onpt.Array[tuple[int, ...], np.floating[Any] | np.integer[Any] | np.bool_]
_Array_1d: TypeAlias = onpt.Array[tuple[int], np.number[Any] | np.bool_]
_Array_1d_i0: TypeAlias = onpt.Array[tuple[int], np.intp]
_Array_1d_f8: TypeAlias = onpt.Array[tuple[int], np.float64]
_Array_2d_f8: TypeAlias = onpt.Array[tuple[int, int], np.float64]

_Args: TypeAlias = tuple[object, ...]
_Brack: TypeAlias = tuple[float, float] | tuple[float, float, float]
_Disp: TypeAlias = Literal[0, 1, 2, 3] | bool | np.bool_
_BracketInfo: TypeAlias = tuple[
    _Scalar_f8, _Scalar_f8, _Scalar_f8,  # xa, xb, xc
    _Scalar_f8, _Scalar_f8, _Scalar_f8,  # fa, fb, fx
    int,  # funcalls
]  # fmt: skip
_WarnFlag: TypeAlias = Literal[0, 1, 2, 3, 4]
_AllVecs: TypeAlias = list[_Array_1d_i0 | _Array_1d_f8]

_XT_contra = TypeVar("_XT_contra", contravariant=True, bound=_Array_1d, default=_Array_1d_f8)
_ValueT_co = TypeVar("_ValueT_co", covariant=True, bound=_Scalar_f, default=_Scalar_f8)
_JacT_co = TypeVar("_JacT_co", covariant=True, bound=_Array_f, default=_Array_1d_f8)

@type_check_only
class _DoesFMin(Protocol):
    def __call__(self, func: _Fn1_1d, x0: _Array_1d_f8, /, *, args: _Args) -> _Array_f: ...

###

# NOTE: Unlike the docs suggest, `OptimizeResult` has no attributes by default, as e.g. `RootResult` has none of these attrs
class OptimizeResult(_RichResult): ...

#
class OptimizeWarning(UserWarning): ...
class BracketError(RuntimeError): ...  # undocumented

# undocumented
class MemoizeJac(Generic[_XT_contra, _ValueT_co, _JacT_co]):
    fun: _Fn1[_XT_contra, tuple[_ValueT_co, _JacT_co]]  # readonly
    x: _XT_contra  # readonly
    _value: _ValueT_co  # readonly
    jac: _JacT_co  # readonly

    def __init__(self, /, fun: _Fn1[_XT_contra, tuple[_ValueT_co, _JacT_co]]) -> None: ...
    def __call__(self, /, x: _XT_contra, *args: object) -> _ValueT_co: ...
    def derivative(self, /, x: _XT_contra, *args: object) -> _JacT_co: ...

# undocumented
class Brent(Generic[_ValueT_co]):
    _mintol: Final[float]  # 1e-11
    _cg: Final[float]  # 0.3819660

    func: _Fn1[np.float64, _ValueT_co]  # `float & np.float64`
    args: Final[_Args]
    tol: Final[float]
    maxiter: Final[int]

    xmin: _Scalar_f8 | None
    fval: _Scalar_f8 | None
    iter: int
    funcalls: int
    disp: _Disp
    brack: _Brack | None  # might be undefined; set by `set_bracket`

    def __init__(
        self,
        /,
        func: _Fn1_0d,
        args: _Args = (),
        tol: AnyReal = 1.48e-08,
        maxiter: int = 500,
        full_output: AnyBool = 0,  # ignored
        disp: _Disp = 0,
    ) -> None: ...
    def set_bracket(self, /, brack: _Brack | None = None) -> None: ...
    def get_bracket_info(self, /) -> _BracketInfo: ...
    def optimize(self, /) -> None: ...
    @overload
    def get_result(self, /, full_output: _Falsy = False) -> _Scalar_f8: ...
    @overload
    def get_result(self, /, full_output: _Truthy) -> tuple[_Scalar_f8, _Scalar_f8, int, int]: ...  # xmin, fval, itere, funcalls

# undocumented
@overload
def is_finite_scalar(x: _Scalar) -> np.bool_: ...
@overload  # returns a `np.ndarray` of `size = 1`, but could have any `ndim`
def is_finite_scalar(x: _Array) -> Literal[False] | onpt.Array[tuple[Literal[1], ...], np.bool_]: ...

# undocumented
@overload
def vecnorm(x: _Scalar, ord: AnyReal = 2) -> AnyReal: ...
@overload
def vecnorm(x: _Array, ord: AnyInt = 2) -> _Array_f_co: ...
@overload
def vecnorm(x: _ArrayLikeNumber_co, ord: AnyInt = 2) -> AnyReal: ...
@overload
def vecnorm(x: _ArrayLikeNumber_co, ord: AnyReal = 2) -> AnyReal | _Array_f_co: ...

# undocumented
def approx_fhess_p(
    x0: _ArrayLikeFloat_co,
    p: AnyReal,
    fprime: _Fn1[_Array_1d_f8, _Array_f_co],
    epsilon: AnyReal | _Array_f_co,  # scalar or 1d ndarray
    *args: object,
) -> _Array_1d_f8: ...

#
@overload  # full_output: False = ..., retall: False = ...
def fmin(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> _Array_1d_f8: ...
@overload  # full_output: False = ..., retall: True (keyword)
def fmin(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    *,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _AllVecs]: ...
@overload  # full_output: True  (keyword), retall: False = ...
def fmin(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, AnyReal, int, int, _WarnFlag]: ...  # x, fun, nit, nfev, status
@overload  # full_output: True  (keyword), retall: True
def fmin(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    initial_simplex: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, AnyReal, int, int, _WarnFlag, _AllVecs]: ...  # x, fun, nit, nfev, status, allvecs

#
@overload  # full_output: False = ..., retall: False = ...
def fmin_bfgs(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    xrtol: AnyReal = 0,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> _Array_1d_f8: ...
@overload  # full_output: False = ..., retall: True  (keyword)
def fmin_bfgs(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    *,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    xrtol: AnyReal = 0,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _AllVecs]: ...
@overload  # full_output: True (keyword), retall: False = ...
def fmin_bfgs(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    xrtol: AnyReal = 0,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _Scalar_f8, _Array_1d_f8, _Array_2d_f8, int, int, _WarnFlag]: ...
@overload  # full_output: True (keyword), retall: True (keyword)
def fmin_bfgs(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    xrtol: AnyReal = 0,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
    hess_inv0: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _Scalar_f8, _Array_1d_f8, _Array_2d_f8, int, int, _WarnFlag, _AllVecs]: ...

#
@overload  # full_output: False = ..., retall: False = ...
def fmin_cg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> _Array_1d_f8: ...
@overload  # full_output: False = ..., retall: True  (keyword)
def fmin_cg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    *,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _AllVecs]: ...
@overload  # full_output: True (keyword), retall: False = ...
def fmin_cg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _Scalar_f8, int, int, _WarnFlag]: ...
@overload  # full_output: True (keyword), retall: True (keyword)
def fmin_cg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    gtol: AnyReal = 1e-05,
    norm: AnyReal = ...,  # inf
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _Scalar_f8, int, int, _WarnFlag, _AllVecs]: ...

# TODO(jorenham): overload `full_output` / `retall`
@overload  # full_output: False = ..., retall: False = ...
def fmin_ncg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co],
    fhess_p: _Fn2[_Array_1d_f8, _Array_1d_f8, _ArrayLikeFloat_co] | None = None,
    fhess: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    avextol: AnyReal = 1e-5,
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> _Array_1d_f8: ...
@overload  # full_output: False = ..., retall: True  (keyword)
def fmin_ncg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co],
    fhess_p: _Fn2[_Array_1d_f8, _Array_1d_f8, _ArrayLikeFloat_co] | None = None,
    fhess: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    avextol: AnyReal = 1e-5,
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    *,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _AllVecs]: ...
@overload  # full_output: True (keyword), retall: False = ...
def fmin_ncg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co],
    fhess_p: _Fn2[_Array_1d_f8, _Array_1d_f8, _ArrayLikeFloat_co] | None = None,
    fhess: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    avextol: AnyReal = 1e-5,
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _Scalar_f8, int, int, int, _WarnFlag]: ...
@overload  # full_output: True (keyword), retall: True (keyword)
def fmin_ncg(
    f: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    fprime: _Fn1_1d[_Array_f_co],
    fhess_p: _Fn2[_Array_1d_f8, _Array_1d_f8, _ArrayLikeFloat_co] | None = None,
    fhess: _Fn1_1d[_Array_f_co] | None = None,
    args: _Args = (),
    avextol: AnyReal = 1e-5,
    epsilon: AnyReal | _Array_f_co = ...,
    maxiter: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    c1: AnyReal = 1e-4,
    c2: AnyReal = 0.9,
) -> tuple[_Array_1d_f8, _Scalar_f8, int, int, int, _WarnFlag, _AllVecs]: ...

#
@overload  # full_output: False = ..., retall: False = ...
def fmin_powell(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> _Array_1d_f8: ...
@overload  # full_output: False = ..., retall: True  (keyword)
def fmin_powell(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
    *,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _AllVecs]: ...
@overload  # full_output: True (keyword), retall: False = ...
def fmin_powell(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Falsy = 0,
    callback: _Callback_1d | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _Scalar_f8, _Array_2d_f8, int, int, _WarnFlag]: ...
@overload  # full_output: True (keyword), retall: True (keyword)
def fmin_powell(
    func: _Fn1_1d,
    x0: _ArrayLikeFloat_co,
    args: _Args = (),
    xtol: AnyReal = 1e-4,
    ftol: AnyReal = 1e-4,
    maxiter: int | None = None,
    maxfun: int | None = None,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
    retall: _Truthy,
    callback: _Callback_1d | None = None,
    direc: _ArrayLikeFloat_co | None = None,
) -> tuple[_Array_1d_f8, _Scalar_f8, _Array_2d_f8, int, int, _WarnFlag, _AllVecs]: ...

#
@overload  # full_output: False = ...
def fminbound(
    func: _Fn1_0d,
    x1: AnyReal,
    x2: AnyReal,
    args: _Args = (),
    xtol: AnyReal = 1e-05,
    maxfun: int = 500,
    full_output: _Falsy = 0,
    disp: _Disp = 1,
) -> _Scalar_f8: ...
@overload  # full_output: True (keyword)
def fminbound(
    func: _Fn1_0d,
    x1: AnyReal,
    x2: AnyReal,
    args: _Args = (),
    xtol: AnyReal = 1e-05,
    maxfun: int = 500,
    *,
    full_output: _Truthy,
    disp: _Disp = 1,
) -> tuple[_Scalar_f8, _Scalar_f8, _WarnFlag, int]: ...  # x, fun, status, nfev

#
@overload  # full_output: False = ...
def brute(
    func: _Fn1_1d,
    ranges: tuple[tuple[AnyReal, AnyReal] | slice, ...],
    args: _Args = (),
    Ns: int = 20,
    full_output: _Falsy = 0,
    finish: _DoesFMin | None = ...,  # default: `fmin`
    disp: AnyBool = False,
    workers: int | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1,
) -> _Array_1d_f8: ...
@overload  # full_output: True (keyword)
def brute(
    func: _Fn1_1d,
    ranges: tuple[tuple[AnyReal, AnyReal] | slice, ...],
    args: _Args = (),
    Ns: int = 20,
    *,
    full_output: _Truthy,
    finish: _DoesFMin | None = ...,  # default: `fmin`
    disp: AnyBool = False,
    workers: int | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1,
) -> tuple[
    _Array_1d_f8,
    np.float64,
    onpt.Array[tuple[Literal[2], int, int], np.float64],
    onpt.Array[tuple[int, int], np.floating[Any]],
]: ...

#
@overload  # full_output: False = ...
def brent(
    func: _Fn1_0d,
    args: _Args = (),
    brack: Brack | None = None,
    tol: AnyReal = 1.48e-08,
    full_output: _Falsy = 0,
    maxiter: int = 500,
) -> _Scalar_f8: ...
@overload  # full_output: True (positional)
def brent(
    func: _Fn1_0d,
    args: _Args,
    brack: Brack | None,
    tol: AnyReal,
    full_output: _Truthy,
    maxiter: int = 500,
) -> tuple[_Scalar_f8, _Scalar_f8, int, int]: ...
@overload  # full_output: True (keyword)
def brent(
    func: _Fn1_0d,
    args: _Args = (),
    brack: Brack | None = None,
    tol: AnyReal = 1.48e-08,
    *,
    full_output: _Truthy,
    maxiter: int = 500,
) -> tuple[_Scalar_f8, _Scalar_f8, int, int]: ...

# TODO(jorenham): overload `full_output`
@overload  # full_output: False = ...
def golden(
    func: _Fn1_0d,
    args: _Args = (),
    brack: Brack | None = None,
    tol: AnyReal = ...,
    full_output: _Falsy = 0,
    maxiter: int = 5_000,
) -> _Scalar_f8: ...
@overload  # full_output: True (positional)
def golden(
    func: _Fn1_0d,
    args: _Args,
    brack: Brack | None,
    tol: AnyReal,
    full_output: _Truthy,
    maxiter: int = 5_000,
) -> tuple[_Scalar_f8, _Scalar_f8, int]: ...
@overload  # full_output: True (keyword)
def golden(
    func: _Fn1_0d,
    args: _Args = (),
    brack: Brack | None = None,
    tol: AnyReal = ...,
    *,
    full_output: _Truthy,
    maxiter: int = 5_000,
) -> tuple[_Scalar_f8, _Scalar_f8, int]: ...

#
def bracket(
    func: _Fn1_0d,
    xa: AnyReal = 0.0,
    xb: AnyReal = 1.0,
    args: _Args = (),
    grow_limit: AnyReal = 110.0,
    maxiter: int = 1_000,
) -> _BracketInfo: ...

# rosenbrock
def rosen(x: _ArrayLikeFloat_co) -> AnyReal: ...
def rosen_der(x: _ArrayLikeFloat_co) -> _Array_1d_f8: ...
def rosen_hess(x: _ArrayLikeFloat_co) -> _Array_2d_f8: ...
def rosen_hess_prod(x: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co) -> _Array_1d_f8: ...

#
@overload  # disp: True = ...
def show_options(solver: Solver | None = None, method: MethodAll | None = None, disp: _Truthy = True) -> None: ...
@overload  # disp: False  (positional)
def show_options(solver: Solver | None, method: MethodAll | None, disp: _Falsy) -> str: ...
@overload  # disp: False  (keyword)
def show_options(solver: Solver | None = None, method: MethodAll | None = None, *, disp: _Falsy) -> str: ...

#
def approx_fprime(
    xk: _ArrayLikeFloat_co,
    f: _Fn1_1d,
    epsilon: AnyReal | _Array_f_co = ...,
    *args: object,
) -> _Array_1d_f8: ...

#
def check_grad(
    func: _Fn1_1d,
    grad: _Fn1_1d[_Array_f_co],
    x0: _ArrayLikeFloat_co,
    *args: object,
    epsilon: AnyReal = ...,
    direction: Literal["all", "random"] = "all",
    seed: Seed | None = None,
) -> _Scalar_f8: ...
