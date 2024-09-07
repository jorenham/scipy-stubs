import collections
from collections.abc import Callable
from typing import Any, Final, Generic, Literal, NoReturn, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import Never, TypeVar, TypeVarTuple, Unpack, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])
_S = TypeVar("_S")
_T = TypeVar("_T")
_VT = TypeVar("_VT", default=object)

_FloatND: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]] | float | np.floating[Any]
_NDT_f = TypeVar("_NDT_f", bound=_FloatND)
_NDT_f_co = TypeVar("_NDT_f_co", bound=_FloatND, covariant=True, default=_FloatND)
_SCT_f_co = TypeVar("_SCT_f_co", bound=np.floating[Any], covariant=True, default=np.float64)

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_NBT = TypeVar("_NBT", bound=npt.NBitBase)

@type_check_only
class _QuadVecFunc(Protocol[_NDT_f_co, Unpack[_Ts]]):
    def __call__(self, x: float, /, *args: Unpack[_Ts]) -> _NDT_f_co: ...

@type_check_only
class _DoesMap(Protocol):
    def __call__(
        self,
        func: Callable[[_S], _T],
        iterable: op.CanIter[op.CanNext[_S]],
        /,
    ) -> op.CanIter[op.CanIterSelf[_T]]: ...

@type_check_only
class _InfiniteFunc(Protocol[_NDT_f_co]):
    def get_t(self, /, x: float) -> float: ...
    def __call__(self, /, t: float) -> _NDT_f_co: ...

class LRUDict(collections.OrderedDict[tuple[float, float], _VT], Generic[_VT]):
    def __init__(self, /, max_size: int) -> None: ...
    @override
    def update(self, other: Never, /) -> NoReturn: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class SemiInfiniteFunc(_InfiniteFunc[_NDT_f_co], Generic[_NDT_f_co]):
    def __init__(self, /, func: Callable[[float], _NDT_f_co], start: float, infty: bool) -> None: ...

class DoubleInfiniteFunc(_InfiniteFunc, Generic[_NDT_f_co]):
    def __init__(self, /, func: Callable[[float], _NDT_f_co]) -> None: ...

# NOTE: This is only used as "info dict" for `quad_vec(..., full_output=True)`,
# even though, confusingly, it is not even even a mapping.
# NOTE: Because this "bunch" is only used as "info dict" (and nowhere else),
# its the ~keys~ attributes have been annotated right here.
class _Bunch(Generic[_SCT_f_co]):
    def __init__(
        self,
        /,
        *,
        success: bool,
        status: Literal[0, 1, 2],
        neval: int,
        message: str,
        intervals: onpt.Array[tuple[int, Literal[2]], np.float64],
        errors: onpt.Array[tuple[int], np.float64],
        integrals: onpt.Array[tuple[int, Literal[2]], _SCT_f_co],
    ) -> None: ...
    success: Final[bool]
    status: Final[Literal[0, 1, 2]]
    neval: Final[int]
    message: Final[str]
    intervals: Final[onpt.Array[tuple[int, Literal[2]], np.float64]]
    errors: Final[onpt.Array[tuple[int], np.float64]]
    integrals: onpt.Array[tuple[int, Literal[2]], _SCT_f_co]

@overload
def quad_vec(
    f: Callable[[float], _NDT_f],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-08,
    norm: Literal["max", "2"] = "2",
    cache_size: float = 100_000_000,
    limit: float = 10_000,
    workers: int | _DoesMap = 1,
    points: op.CanIter[op.CanNext[float]] | None = None,
    quadrature: Literal["gk21", "gk15", "trapezoid"] | None = None,
    full_output: Literal[False, 0, None] = False,
    *,
    args: tuple[()] = ...,
) -> tuple[_NDT_f, float]: ...
@overload
def quad_vec(
    f: _QuadVecFunc[_NDT_f, Unpack[_Ts]],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-08,
    norm: Literal["max", "2"] = "2",
    cache_size: float = 100_000_000,
    limit: float = 10_000,
    workers: int | _DoesMap = 1,
    points: op.CanIter[op.CanNext[float]] | None = None,
    quadrature: Literal["gk21", "gk15", "trapezoid"] | None = None,
    full_output: Literal[False, 0, None] = False,
    *,
    args: tuple[Unpack[_Ts]],
) -> tuple[_NDT_f, float]: ...
# false positive (basedmypy==1.6.0 / mypy==1.11.1)
@overload
def quad_vec(  # type: ignore[overload-overlap]
    f: _QuadVecFunc[float | np.float64, Unpack[_Ts]],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-08,
    norm: Literal["max", "2"] = "2",
    cache_size: float = 100_000_000,
    limit: float = 10_000,
    workers: int | _DoesMap = 1,
    points: op.CanIter[op.CanNext[float]] | None = None,
    quadrature: Literal["gk21", "gk15", "trapezoid"] | None = None,
    *,
    full_output: Literal[True, 1],
    args: tuple[Unpack[_Ts]] = ...,
) -> tuple[float | np.float64, float, _Bunch[np.float64]]: ...
@overload
def quad_vec(
    f: _QuadVecFunc[np.floating[_NBT], Unpack[_Ts]],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-08,
    norm: Literal["max", "2"] = "2",
    cache_size: float = 100_000_000,
    limit: float = 10_000,
    workers: int | _DoesMap = 1,
    points: op.CanIter[op.CanNext[float]] | None = None,
    quadrature: Literal["gk21", "gk15", "trapezoid"] | None = None,
    *,
    full_output: Literal[True, 1],
    args: tuple[Unpack[_Ts]] = ...,
) -> tuple[np.floating[_NBT], float, _Bunch[np.floating[_NBT]]]: ...
@overload
def quad_vec(
    f: _QuadVecFunc[onpt.Array[_ShapeT, np.floating[_NBT]], Unpack[_Ts]],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-08,
    norm: Literal["max", "2"] = "2",
    cache_size: float = 100_000_000,
    limit: float = 10_000,
    workers: int | _DoesMap = 1,
    points: op.CanIter[op.CanNext[float]] | None = None,
    quadrature: Literal["gk21", "gk15", "trapezoid"] | None = None,
    *,
    full_output: Literal[True, 1],
    args: tuple[Unpack[_Ts]] = ...,
) -> tuple[onpt.Array[_ShapeT, np.floating[_NBT]], float, _Bunch[np.floating[_NBT]]]: ...
