from typing import NamedTuple, TypeVar

import numpy as np

from scipy._lib._array_api import array_namespace as array_namespace, is_numpy as is_numpy, xp_size as xp_size
from scipy._typing import Untyped

AxisError: type[Exception]
ComplexWarning: type[Warning]
VisibleDeprecationWarning: type[Warning]
DTypePromotionError = TypeError
np_long: type
np_ulong: type
IntNumber: Untyped
DecimalNumber: Untyped
copy_if_needed: bool | None
SeedType: Untyped
GeneratorType = TypeVar("GeneratorType", bound=np.random.Generator | np.random.RandomState)

class Generator: ...

def float_factorial(n: int) -> float: ...
def check_random_state(seed) -> Untyped: ...

class FullArgSpec(NamedTuple):
    args: Untyped
    varargs: Untyped
    varkw: Untyped
    defaults: Untyped
    kwonlyargs: Untyped
    kwonlydefaults: Untyped
    annotations: Untyped

def getfullargspec_no_self(func) -> Untyped: ...

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
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None
    ): ...
    def __call__(self, func, iterable) -> Untyped: ...

def rng_integers(
    gen, low, high: Untyped | None = None, size: Untyped | None = None, dtype: str = "int64", endpoint: bool = False
) -> Untyped: ...
def normalize_axis_index(axis, ndim) -> Untyped: ...

class _RichResult(dict):
    def __getattr__(self, name) -> Untyped: ...
    __setattr__: Untyped
    __delattr__: Untyped
    def __dir__(self) -> Untyped: ...
