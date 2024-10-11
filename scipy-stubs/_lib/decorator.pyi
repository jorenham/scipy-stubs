from contextlib import _GeneratorContextManager
from collections.abc import Callable, Iterator
from typing import Final, Generic, ParamSpec
from typing_extensions import TypeVar

from scipy._typing import Untyped, UntypedCallable

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True, default=object)
_Tss = ParamSpec("_Tss")

__version__: Final[str] = ...

class ContextManager(_GeneratorContextManager[_T_co], Generic[_T_co]):
    def __init__(self, g: Callable[_Tss, Iterator[_T_co]], *a: _Tss.args, **k: _Tss.kwargs) -> None: ...

def decorate(func: UntypedCallable, caller: Untyped) -> Untyped: ...
def decorator(caller: Untyped, _func: UntypedCallable | None = None) -> UntypedCallable: ...
def contextmanager(g: Callable[_Tss, Iterator[_T]], *a: _Tss.args, **k: _Tss.kwargs) -> Callable[[], ContextManager[_T]]: ...
