import ctypes as ct
from _ctypes import CFuncPtr as PyCFuncPtr
from types import ModuleType
from typing import ClassVar, Generic, Literal, TypeAlias
from typing_extensions import CapsuleType, Self, TypeVar

from scipy._typing import Untyped

__all__ = ["LowLevelCallable"]

# TODO: add `types-cffi` as dependency
CFFIFuncP: TypeAlias = Untyped
CFFIVoidP: TypeAlias = Untyped
ffi: Literal[False] | None

class CData: ...

_Function: TypeAlias = CapsuleType | PyCFuncPtr | CFFIFuncP | CData
_UserData: TypeAlias = CapsuleType | ct.c_void_p | CFFIVoidP

_FuncT_co = TypeVar("_FuncT_co", bound=_Function, covariant=True, default=_Function)
_DataT_co = TypeVar("_DataT_co", bound=_UserData | None, covariant=True, default=None)

class LowLevelCallable(tuple[CapsuleType, _FuncT_co, _DataT_co], Generic[_FuncT_co, _DataT_co]):
    # NOTE: __getitem__ will raise a ValueError
    __getitem__: ClassVar[None]  # type: ignore[assignment]

    def __new__(
        cls,
        /,
        function: _FuncT_co | LowLevelCallable[_FuncT_co, _DataT_co],
        user_data: Untyped | None = None,
        signature: str | None = None,
    ) -> Self: ...
    @property
    def function(self, /) -> _FuncT_co: ...
    @property
    def user_data(self, /) -> _DataT_co: ...
    @property
    def signature(self, /) -> str: ...
    @classmethod
    def from_cython(
        cls,
        /,
        module: ModuleType,
        name: str,
        user_data: _UserData | None = None,
        signature: str | None = None,
    ) -> Untyped: ...
