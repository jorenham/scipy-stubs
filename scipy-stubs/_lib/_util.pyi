from multiprocessing.pool import Pool
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Concatenate, Final, Generic, NamedTuple, TypeAlias, overload
from typing_extensions import TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeInt
from numpy.random import Generator as Generator  # noqa: ICN003
from scipy._typing import RNG, EnterSelfMixin, Seed

_T = TypeVar("_T", default=object)
_T_co = TypeVar("_T_co", covariant=True, default=object)
_T_contra = TypeVar("_T_contra", contravariant=True, default=object)
_VT = TypeVar("_VT")
_RT = TypeVar("_RT")
_AxisT = TypeVar("_AxisT", bound=int | np.integer[Any])

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
    axis: None | int
    ndim: None | int
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
    pool: int | Pool | None
    def __init__(self, /, pool: int | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1) -> None: ...
    def __call__(self, func: Callable[[_VT], _RT], iterable: Iterable[_VT]) -> Sequence[_RT]: ...
    def terminate(self, /) -> None: ...
    def join(self, /) -> None: ...
    def close(self, /) -> None: ...

class _RichResult(dict[str, _T]):
    def __getattr__(self, name: str, /) -> _T: ...
    @override
    def __setattr__(self, name: str, value: _T, /) -> None: ...

def float_factorial(n: int) -> float: ...
def getfullargspec_no_self(func: Callable[..., object]) -> FullArgSpec: ...
def check_random_state(seed: Seed) -> np.random.Generator | np.random.RandomState: ...
def rng_integers(
    gen: RNG | None,
    low: _ArrayLikeInt,
    high: _ArrayLikeInt | None = None,
    size: Sequence[int] | None = None,
    dtype: onpt.AnyIntegerDType = "int64",
    endpoint: bool = False,
) -> np.integer[Any] | npt.NDArray[np.integer[Any]]: ...
def normalize_axis_index(axis: _AxisT, ndim: int | np.integer[Any]) -> _AxisT: ...
