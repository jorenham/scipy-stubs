from collections.abc import Callable
from types import TracebackType
from typing import Any
from typing_extensions import TypeVar

from scipy._typing import Untyped

__all__ = ["ReentrancyError", "ReentrancyLock", "non_reentrant"]

_FT = TypeVar("_FT", bound=Callable[..., Any])

class ReentrancyError(RuntimeError): ...

class ReentrancyLock:
    def __init__(self, /, err_msg: str) -> None: ...
    def __enter__(self, /) -> None: ...
    def __exit__(
        self,
        /,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def decorate(self, func: _FT, /) -> _FT: ...

def non_reentrant(err_msg: Untyped | None = None) -> Untyped: ...
