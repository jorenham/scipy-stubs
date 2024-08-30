from types import TracebackType
from typing import Any, NamedTuple, TypeAlias
from typing_extensions import TypeVar, override

import numpy as np
from scipy._typing import Untyped

AxisError: type[Exception]
ComplexWarning: type[Warning]
VisibleDeprecationWarning: type[Warning]
DTypePromotionError = TypeError
np_long: type
np_ulong: type
copy_if_needed: bool | None

IntNumber: TypeAlias = int | np.integer[Any]
DecimalNumber: TypeAlias = float | np.floating[Any] | np.integer[Any]
SeedType: TypeAlias = IntNumber | np.random.Generator | np.random.RandomState | None

class Generator: ...

def float_factorial(n: int) -> float: ...
def check_random_state(seed: SeedType) -> np.random.Generator | np.random.RandomState: ...

class FullArgSpec(NamedTuple):
    args: Untyped
    varargs: Untyped
    varkw: Untyped
    defaults: Untyped
    kwonlyargs: Untyped
    kwonlydefaults: Untyped
    annotations: Untyped

def getfullargspec_no_self(func) -> FullArgSpec: ...

class _FunctionWrapper:
    f: Untyped
    args: Untyped
    def __init__(self, f, args) -> None: ...
    def __call__(self, x) -> Untyped: ...

class MapWrapper:
    pool: Untyped
    def __init__(self, pool: int = 1): ...
    def __enter__(self) -> Untyped: ...
    def terminate(self): ...
    def join(self): ...
    def close(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ): ...
    def __call__(self, func, iterable) -> Untyped: ...

def rng_integers(
    gen,
    low,
    high: Untyped | None = None,
    size: Untyped | None = None,
    dtype: str = "int64",
    endpoint: bool = False,
) -> Untyped: ...
def normalize_axis_index(axis, ndim) -> Untyped: ...

_VT = TypeVar("_VT", default=object)

class _RichResult(dict[str, _VT]):
    def __getattr__(self, name: str, /) -> _VT: ...
    @override
    def __setattr__(self, name: str, value: _VT, /) -> None: ...
    @override
    def __delattr__(self, name: str, /) -> None: ...
    @override
    def __dir__(self, /) -> list[str]: ...
