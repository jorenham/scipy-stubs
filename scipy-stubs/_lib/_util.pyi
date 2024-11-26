import multiprocessing.pool as mpp
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Concatenate, Final, Generic, NamedTuple, TypeAlias, overload
from typing_extensions import TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import RNG, EnterSelfMixin

_AnyRNG = TypeVar("_AnyRNG", np.random.RandomState, np.random.Generator)

_T = TypeVar("_T", default=object)
_T_co = TypeVar("_T_co", covariant=True, default=object)
_T_contra = TypeVar("_T_contra", contravariant=True, default=object)
_VT = TypeVar("_VT")
_RT = TypeVar("_RT")
_AxisT = TypeVar("_AxisT", bound=np.integer[Any])

###

np_long: Final[type[np.int32 | np.int64]] = ...
np_ulong: Final[type[np.uint32 | np.uint64]] = ...
copy_if_needed: Final[bool | None] = ...

IntNumber: TypeAlias = int | np.integer[Any]
DecimalNumber: TypeAlias = float | np.floating[Any] | np.integer[Any]

class ComplexWarning(RuntimeWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class DTypePromotionError(TypeError): ...

class AxisError(ValueError, IndexError):
    _msg: Final[str | None]
    axis: Final[int | None]
    ndim: Final[int | None]

    @overload
    def __init__(self, /, axis: str, ndim: None = None, msg_prefix: None = None) -> None: ...
    @overload
    def __init__(self, /, axis: int, ndim: int, msg_prefix: str | None = None) -> None: ...

class FullArgSpec(NamedTuple):
    args: Sequence[str]
    varargs: str | None
    varkw: str | None
    defaults: tuple[object, ...] | None
    kwonlyargs: Sequence[str]
    kwonlydefaults: Mapping[str, object] | None
    annotations: Mapping[str, type | object | str]

class _FunctionWrapper(Generic[_T_contra, _T_co]):
    f: Callable[Concatenate[_T_contra, ...], _T_co]
    args: tuple[object, ...]

    @overload
    def __init__(self, /, f: Callable[[_T_contra], _T_co], args: tuple[()]) -> None: ...
    @overload
    def __init__(self, /, f: Callable[Concatenate[_T_contra, ...], _T_co], args: tuple[object, ...]) -> None: ...
    def __call__(self, /, x: _T_contra) -> _T_co: ...

class MapWrapper(EnterSelfMixin):
    pool: int | mpp.Pool | None

    def __init__(self, /, pool: Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] | int = 1) -> None: ...
    def __call__(self, /, func: Callable[[_VT], _RT], iterable: Iterable[_VT]) -> Sequence[_RT]: ...
    def terminate(self, /) -> None: ...
    def join(self, /) -> None: ...
    def close(self, /) -> None: ...

class _RichResult(dict[str, _T]):
    def __getattr__(self, name: str, /) -> _T: ...
    @override
    def __setattr__(self, name: str, value: _T, /) -> None: ...

#
def float_factorial(n: int) -> float: ...

#
def getfullargspec_no_self(func: Callable[..., object]) -> FullArgSpec: ...

#
@overload
def check_random_state(seed: _AnyRNG) -> _AnyRNG: ...
@overload
def check_random_state(seed: int | np.integer[Any] | types.ModuleType | None) -> np.random.RandomState: ...

#
@overload
def rng_integers(
    gen: RNG | None,
    low: onp.ToInt | onp.ToIntND,
    high: onp.ToInt | onp.ToIntND | None = None,
    size: tuple[()] | None = None,
    dtype: onp.AnyIntegerDType = "int64",
    endpoint: op.CanBool = False,
) -> np.integer[Any]: ...
@overload
def rng_integers(
    gen: RNG | None,
    low: onp.ToInt | onp.ToIntND,
    high: onp.ToInt | onp.ToIntND | None = None,
    size: op.CanIndex | Sequence[op.CanIndex] | None = None,
    dtype: onp.AnyIntegerDType = "int64",
    endpoint: op.CanBool = False,
) -> np.integer[Any] | onp.ArrayND[np.integer[Any]]: ...

#
@overload
def normalize_axis_index(axis: int, ndim: int) -> int: ...
@overload
def normalize_axis_index(axis: int, ndim: _AxisT) -> _AxisT: ...
@overload
def normalize_axis_index(axis: _AxisT, ndim: int | _AxisT) -> _AxisT: ...
