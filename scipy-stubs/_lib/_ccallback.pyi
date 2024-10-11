import ctypes as ct
from _ctypes import CFuncPtr as PyCFuncPtr
from types import ModuleType
from typing import Generic, Literal, NoReturn, TypeAlias
from typing_extensions import CapsuleType, Never, Self, TypeVar, override

from cffi.model import FunctionPtrType as _CFFIFuncP, PointerType as _CFFIVoidP
from scipy._typing import Untyped

_Function: TypeAlias = CapsuleType | PyCFuncPtr | _CFFIFuncP | CData
_UserData: TypeAlias = CapsuleType | ct.c_void_p | _CFFIVoidP

_FuncT_co = TypeVar("_FuncT_co", bound=_Function, covariant=True, default=_Function)
_DataT_co = TypeVar("_DataT_co", bound=_UserData | None, covariant=True, default=_UserData)

ffi: Literal[False] | None

class CData: ...

class LowLevelCallable(tuple[CapsuleType, _FuncT_co, _DataT_co], Generic[_FuncT_co, _DataT_co]):
    @property
    def function(self, /) -> _FuncT_co: ...
    @property
    def user_data(self, /) -> _DataT_co: ...
    @property
    def signature(self, /) -> str: ...
    def __new__(
        cls,
        function: _FuncT_co | LowLevelCallable[_FuncT_co, _DataT_co],
        user_data: Untyped | None = None,
        signature: str | None = None,
    ) -> Self: ...
    @classmethod
    def from_cython(
        cls,
        module: ModuleType,
        name: str,
        user_data: _UserData | None = None,
        signature: str | None = None,
    ) -> Self: ...

    # NOTE: `__getitem__` will always raise a `ValueError`
    @override
    def __getitem__(self, idx: Never, /) -> NoReturn: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
