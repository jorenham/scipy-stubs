from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar

from scipy._typing import Untyped

__all__ = ["_deprecated"]

_F = TypeVar("_F", bound=Callable[..., Any])

class _DeprecationHelperStr:
    def __init__(self, content: Untyped, message: str) -> None: ...

def _deprecated(msg: str, stacklevel: int = 2) -> Callable[[_F], _F]: ...
def deprecate_cython_api(
    module: ModuleType,
    routine_name: str,
    new_name: str | None = None,
    message: str | None = None,
) -> None: ...
