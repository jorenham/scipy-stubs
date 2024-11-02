from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, final, type_check_only
from typing_extensions import TypeVar

from optype import CanWith

_RT_co = TypeVar("_RT_co", covariant=True, default=Any)

@type_check_only
class _BaseBackend(Protocol[_RT_co]):
    __ua_domain__: ClassVar = "numpy.scipy.fft"
    @staticmethod
    def __ua_function__(method: str, args: Sequence[object], kwargs: Mapping[str, object]) -> _RT_co: ...

###

@final
class _ScipyBackend(_BaseBackend): ...

def set_global_backend(backend: _BaseBackend, coerce: bool = False, only: bool = False, try_last: bool = False) -> None: ...
def register_backend(backend: _BaseBackend) -> None: ...
def set_backend(backend: _BaseBackend, coerce: bool = False, only: bool = False) -> CanWith[None]: ...
def skip_backend(backend: _BaseBackend) -> CanWith[None]: ...
