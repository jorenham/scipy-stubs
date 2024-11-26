from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar, TypeVarTuple, Unpack, deprecated

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.stats.qmc import QMCEngine

__all__ = [
    "AccuracyWarning",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "fixed_quad",
    "newton_cotes",
    "qmc_quad",
    "quadrature",
    "romb",
    "romberg",
    "simpson",
    "trapezoid",
]

_QuadFuncOut: TypeAlias = onp.ArrayND[np.floating[Any]] | Sequence[float]

_NDT_f = TypeVar("_NDT_f", bound=_QuadFuncOut)
_NDT_f_co = TypeVar("_NDT_f_co", bound=_QuadFuncOut, covariant=True)
_Ts = TypeVarTuple("_Ts")

@type_check_only
class _VectorizedQuadFunc(Protocol[_NDT_f_co, Unpack[_Ts]]):
    def __call__(self, x: onp.Array1D[np.float64], /, *args: Unpack[_Ts]) -> _NDT_f_co: ...

class AccuracyWarning(Warning): ...

class QMCQuadResult(NamedTuple):
    integral: float
    standard_error: float

# sample-based integration
@overload
def trapezoid(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def trapezoid(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...
@overload
def simpson(
    y: onp.ToFloatND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def simpson(
    y: onp.ToComplexND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...
@overload
def romb(
    y: onp.ToFloatND,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def romb(
    y: onp.ToComplexND,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...

# sample-based cumulative integration
@overload
def cumulative_trapezoid(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def cumulative_trapezoid(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[np.inexact[Any]]: ...
@overload
def cumulative_simpson(
    y: onp.ToFloatND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToFloatND | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def cumulative_simpson(
    y: onp.ToComplexND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToComplexND | None = None,
) -> onp.ArrayND[np.inexact[Any]]: ...

# function-based
@overload
def fixed_quad(
    func: _VectorizedQuadFunc[_NDT_f, Unpack[_Ts]],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[Unpack[_Ts]],
    n: op.CanIndex = 5,
) -> _NDT_f: ...
@overload
def fixed_quad(
    # func: _VectorizedQuadFunc[_NDT_f],
    func: Callable[[onp.Array1D[np.float64]], _NDT_f],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[()] = (),
    n: op.CanIndex = 5,
) -> _NDT_f: ...
def qmc_quad(
    func: Callable[[onp.Array2D[np.float64]], onp.ArrayND[np.floating[Any]]],
    a: onp.ToFloat1D,
    b: onp.ToFloat1D,
    *,
    n_estimates: int = 8,
    n_points: int = 1024,
    qrng: QMCEngine | None = None,
    log: bool = False,
) -> QMCQuadResult: ...
@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def quadrature(
    func: object,
    a: object,
    b: object,
    args: tuple[object, ...] = (),
    tol: float = ...,
    rtol: float = ...,
    maxiter: int = ...,
    vec_func: bool = ...,
    miniter: int = ...,
) -> tuple[float, float]: ...
@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def romberg(
    function: object,
    a: object,
    b: object,
    args: tuple[object, ...] = (),
    tol: float = ...,
    rtol: float = ...,
    show: bool = ...,
    divmax: int = ...,
    vec_func: bool = ...,
) -> float: ...

# low-level
def newton_cotes(rn: int, equal: int = 0) -> tuple[onp.Array1D[np.float64], float]: ...
