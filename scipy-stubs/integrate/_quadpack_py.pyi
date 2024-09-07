from collections.abc import Callable
from typing import Any, Final, Generic, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import TypeVar, TypeVarTuple, Unpack

import numpy as np
import numpy.typing as npt
import optype as op
import scipy._typing as spt
from scipy._lib._ccallback import LowLevelCallable
from ._typing import QuadInfoDict, QuadOpts, QuadWeights

__all__ = ["IntegrationWarning", "dblquad", "nquad", "quad", "tplquad"]

_Ts = TypeVarTuple("_Ts")
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_f_contra = TypeVar("_T_f_contra", contravariant=True, default=float)

class IntegrationWarning(UserWarning): ...

# the order of type-params is awkward, but is required because of the TypeVarTuple default.
@type_check_only
class _QuadFunc1(Protocol[_T_co, Unpack[_Ts]]):
    def __call__(self, x: float, /, *args: Unpack[_Ts]) -> _T_co: ...

@type_check_only
class _QuadFunc2R(Protocol[Unpack[_Ts]]):
    def __call__(self, x: float, y: float, /, *args: Unpack[_Ts]) -> spt.AnyReal: ...

@type_check_only
class _QuadFunc3R(Protocol[Unpack[_Ts]]):
    def __call__(self, x: float, y: float, z: float, /, *args: Unpack[_Ts]) -> spt.AnyReal: ...

@type_check_only
class _QuadOutput1C_1(TypedDict):
    real: tuple[QuadInfoDict]
    imag: tuple[QuadInfoDict]

@type_check_only
class _QuadOutput1C_2(TypedDict):
    real: tuple[QuadInfoDict, str]
    imag: tuple[QuadInfoDict, str]

@type_check_only
class _QuadOutput1C_3(TypedDict):
    real: tuple[QuadInfoDict, str, _QuadExplain]
    imag: tuple[QuadInfoDict, str, _QuadExplain]

@type_check_only
class _QuadOutputNC(TypedDict):
    neval: int

_QuadComplexFullOutput: TypeAlias = _QuadOutput1C_1 | _QuadOutput1C_2 | _QuadOutput1C_3
_QuadExplain = TypedDict("_QuadExplain", {0: str, 1: str, 2: str, 3: str, 4: str, 5: str})  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]

@type_check_only
class _CanLenAndIter(Protocol[_T_co]):
    def __len__(self, /) -> int: ...
    def __iter__(self, /) -> op.CanNext[_T_co]: ...

_SizedIterable: TypeAlias = _CanLenAndIter[_T] | op.CanSequence[int, _T]
_QuadRange: TypeAlias = _SizedIterable[float]
_RangeT = TypeVar("_RangeT", bound=_QuadRange, default=_QuadRange)
_RangeT_co = TypeVar("_RangeT_co", bound=_QuadRange, covariant=True, default=_QuadRange)

@type_check_only
class _RangeCallable(Protocol[_T_f_contra, _RangeT_co]):
    def __call__(self, /, *args: _T_f_contra) -> _RangeT_co: ...

_OptT = TypeVar("_OptT", bound=QuadOpts, default=QuadOpts)
_OptT_co = TypeVar("_OptT_co", bound=QuadOpts, covariant=True, default=QuadOpts)

@type_check_only
class _OptCallable(Protocol[_T_f_contra, _OptT_co]):
    def __call__(self, /, *args: _T_f_contra) -> _OptT_co: ...

# 1-dimensional quadrature
@overload
def quad(
    func: _QuadFunc1[spt.AnyReal, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[Unpack[_Ts]],
    full_output: Literal[False, 0, None] = 0,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: Literal[False, 0, None] = False,
) -> tuple[float, float]: ...
@overload
def quad(
    func: _QuadFunc1[spt.AnyReal, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[Unpack[_Ts]],
    full_output: Literal[True, 1],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: Literal[False, 0] = False,
) -> (
    tuple[float, float, QuadInfoDict]
    | tuple[float, float, QuadInfoDict, str]
    | tuple[float, float, QuadInfoDict, str, _QuadExplain]
): ...
@overload
def quad(
    func: _QuadFunc1[spt.AnyComplex, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[Unpack[_Ts]],
    full_output: Literal[False, 0, None] = 0,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    *,
    complex_func: Literal[True, 1],
) -> tuple[complex, complex]: ...
@overload
def quad(
    func: _QuadFunc1[spt.AnyComplex, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[Unpack[_Ts]],
    full_output: Literal[True, 1],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    *,
    complex_func: Literal[True, 1],
) -> tuple[complex, complex, _QuadComplexFullOutput]: ...
@overload
def quad(
    func: Callable[[float], spt.AnyReal] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[()] = (),
    full_output: Literal[False, 0, None] = 0,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: Literal[False, 0, None] = False,
) -> tuple[float, float]: ...
@overload
def quad(
    func: Callable[[float], spt.AnyReal] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[()] = (),
    *,
    full_output: Literal[True, 1],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: Literal[False, 0] = False,
) -> (
    tuple[float, float, QuadInfoDict]
    | tuple[float, float, QuadInfoDict, str]
    | tuple[float, float, QuadInfoDict, str, _QuadExplain]
): ...
@overload
def quad(
    func: Callable[[float], spt.AnyComplex] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[()] = (),
    full_output: Literal[False, 0, None] = 0,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    *,
    complex_func: Literal[True, 1],
) -> tuple[complex, complex]: ...
@overload
def quad(
    func: Callable[[float], spt.AnyComplex] | LowLevelCallable,
    a: float,
    b: float,
    args: tuple[()] = (),
    *,
    full_output: Literal[True, 1],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: op.CanGetitem[int, spt.AnyReal] | None = None,
    weight: QuadWeights | None = None,
    wvar: float | tuple[float, float] | None = None,
    wopts: tuple[int, npt.NDArray[np.float32 | np.float64]] | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: Literal[True, 1],
) -> tuple[complex, complex, _QuadComplexFullOutput]: ...

# 2-dimensional quadrature
@overload
def dblquad(
    func: _QuadFunc2R[spt.AnyReal, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    gfun: float | Callable[[float], float],
    hfun: float | Callable[[float], float],
    args: tuple[Unpack[_Ts]],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
) -> tuple[float, float]: ...
@overload
def dblquad(
    func: Callable[[float, float], spt.AnyReal] | LowLevelCallable,
    a: float,
    b: float,
    gfun: float | Callable[[float], float],
    hfun: float | Callable[[float], float],
    args: tuple[()] = (),
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
) -> tuple[float, float]: ...

# 3-dimensional quadrature
@overload
def tplquad(
    func: _QuadFunc3R[spt.AnyReal, Unpack[_Ts]] | LowLevelCallable,
    a: float,
    b: float,
    gfun: float | Callable[[float], float],
    hfun: float | Callable[[float], float],
    qfun: float | Callable[[float, float], float],
    rfun: float | Callable[[float, float], float],
    args: tuple[Unpack[_Ts]],
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
) -> tuple[float, float]: ...
@overload
def tplquad(
    func: Callable[[float, float, float], spt.AnyReal] | LowLevelCallable,
    a: float,
    b: float,
    gfun: float | Callable[[float], float],
    hfun: float | Callable[[float], float],
    qfun: float | Callable[[float, float], float],
    rfun: float | Callable[[float, float], float],
    args: tuple[()] = (),
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
) -> tuple[float, float]: ...

# N-dimensional quadrature
@overload
def nquad(
    func: Callable[..., float | np.floating[Any]] | LowLevelCallable,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float]],
    args: op.CanIter[op.CanNext[object]] | None = None,
    opts: QuadOpts | Callable[..., QuadOpts] | op.CanIter[op.CanNext[QuadOpts | Callable[..., QuadOpts]]] | None = None,
    full_output: Literal[False, 0, None] = False,
) -> tuple[float, float]: ...
@overload
def nquad(
    func: Callable[..., float | np.floating[Any]] | LowLevelCallable,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float]],
    args: op.CanIter[op.CanNext[object]] | None,
    opts: QuadOpts | _OptCallable | op.CanIter[op.CanNext[QuadOpts | _OptCallable]] | None,
    full_output: Literal[True, 1],
) -> tuple[float, float, _QuadOutputNC]: ...
@overload
def nquad(
    func: Callable[..., float | np.floating[Any]] | LowLevelCallable,
    ranges: _SizedIterable[_QuadRange | _RangeCallable[float]],
    args: op.CanIter[op.CanNext[object]] | None = None,
    opts: QuadOpts | _OptCallable | op.CanIter[op.CanNext[QuadOpts | _OptCallable]] | None = None,
    *,
    full_output: Literal[True, 1],
) -> tuple[float, float, _QuadOutputNC]: ...

class _RangeFunc(_RangeCallable[_T_f_contra, _RangeT], Generic[_T_f_contra, _RangeT]):
    range_: _RangeT
    def __init__(self, /, range_: _RangeT) -> None: ...

class _OptFunc(_OptCallable[_T_f_contra, _OptT], Generic[_T_f_contra, _OptT]):
    opt: _OptT
    def __init__(self, /, opt: _OptT) -> None: ...

_BT_co = TypeVar("_BT_co", bound=bool, covariant=True, default=bool)

class _NQuad(Generic[_BT_co]):
    abserr: Final[float]
    maxdepth: Final[int]
    out_dict: Final[_QuadOutputNC]
    func: Callable[..., float | np.floating[Any]]
    ranges: list[_RangeFunc]
    opts: list[_OptFunc]
    full_output: _BT_co
    def __init__(
        self,
        /,
        func: Callable[..., float | np.floating[Any]],
        ranges: list[_RangeFunc],
        opts: list[_OptFunc],
        full_output: _BT_co,
    ) -> None: ...
    @overload
    def integrate(self: _NQuad[Literal[False]], /, *args: object) -> tuple[float, float]: ...
    @overload
    def integrate(self: _NQuad[Literal[True]], /, *args: object) -> tuple[float, float, _QuadOutputNC]: ...
