from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar, TypeVarTuple, Unpack, deprecated

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
import scipy._typing as spt
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

_NestedReal: TypeAlias = Sequence[float] | Sequence[_NestedReal]
_NestedComplex: TypeAlias = Sequence[complex] | Sequence[_NestedComplex]
_ArrayLike_uif: TypeAlias = _NestedReal | onpt.AnyFloatingArray | onpt.AnyIntegerArray
_ArrayLike_uifc: TypeAlias = _NestedComplex | onpt.AnyNumberArray
_QuadFuncOut: TypeAlias = onpt.Array[tuple[int, ...], np.floating[Any]] | Sequence[float]

_NDT_f = TypeVar("_NDT_f", bound=_QuadFuncOut)
_NDT_f_co = TypeVar("_NDT_f_co", bound=_QuadFuncOut, covariant=True)
_Ts = TypeVarTuple("_Ts")

@type_check_only
class _VectorizedQuadFunc(Protocol[_NDT_f_co, Unpack[_Ts]]):
    def __call__(self, x: onpt.Array[tuple[int], np.float64], /, *args: Unpack[_Ts]) -> _NDT_f_co: ...

class AccuracyWarning(Warning): ...

class QMCQuadResult(NamedTuple):
    integral: float
    standard_error: float

# sample-based integration
@overload
def trapezoid(
    y: _ArrayLike_uif,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def trapezoid(
    y: _ArrayLike_uifc,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | npt.NDArray[np.inexact[Any]]: ...
@overload
def simpson(
    y: _ArrayLike_uif,
    *,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def simpson(
    y: _ArrayLike_uifc,
    *,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | npt.NDArray[np.inexact[Any]]: ...
@overload
def romb(
    y: _ArrayLike_uif,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
@overload
def romb(
    y: _ArrayLike_uifc,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.inexact[Any] | npt.NDArray[np.inexact[Any]]: ...

# sample-based cumulative integration
@overload
def cumulative_trapezoid(
    y: _ArrayLike_uif,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def cumulative_trapezoid(
    y: _ArrayLike_uifc,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> npt.NDArray[np.inexact[Any]]: ...
@overload
def cumulative_simpson(
    y: _ArrayLike_uif,
    *,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    initial: _ArrayLike_uif | None = None,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def cumulative_simpson(
    y: _ArrayLike_uifc,
    *,
    x: _ArrayLike_uif | None = None,
    dx: spt.AnyReal = 1.0,
    axis: op.CanIndex = -1,
    initial: _ArrayLike_uifc | None = None,
) -> npt.NDArray[np.inexact[Any]]: ...

# function-based
@overload
def fixed_quad(
    func: _VectorizedQuadFunc[_NDT_f, Unpack[_Ts]],
    a: spt.AnyReal,
    b: spt.AnyReal,
    args: tuple[Unpack[_Ts]],
    n: op.CanIndex = 5,
) -> _NDT_f: ...
@overload
def fixed_quad(
    # func: _VectorizedQuadFunc[_NDT_f],
    func: Callable[[onpt.Array[tuple[int], np.float64]], _NDT_f],
    a: spt.AnyReal,
    b: spt.AnyReal,
    args: tuple[()] = (),
    n: op.CanIndex = 5,
) -> _NDT_f: ...

def qmc_quad(
    func: Callable[[onpt.Array[tuple[int, int], np.float64]], npt.NDArray[np.floating[Any]]],
    a: Sequence[spt.AnyReal] | onpt.AnyFloatingArray,
    b: Sequence[spt.AnyReal] | onpt.AnyFloatingArray,
    *,
    n_estimates: int = 8,
    n_points: int = 1024,
    qrng: QMCEngine | None = None,
    log: bool = False,
) -> QMCQuadResult: ...
@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def quadrature(*args: object, **kwargs: object) -> tuple[float, float]: ...
@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def romberg(*args: object, **kwargs: object) -> float: ...

# low-level
def newton_cotes(rn: int, equal: int = 0) -> tuple[onpt.Array[tuple[int], np.float64], float]: ...
