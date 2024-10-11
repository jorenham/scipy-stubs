from types import ModuleType
from typing import Literal, Protocol, TypeAlias, TypeVar, type_check_only

import numpy.typing as npt

__all__ = ["_asarray", "array_namespace", "device", "size"]

_DeviceT_co = TypeVar("_DeviceT_co", covariant=True)

# TODO: To be changed to a `Protocol` (once they learn about Python typing, so don't get your hopes up)
# https://github.com/data-apis/array-api/pull/589
Array: TypeAlias = object
ArrayLike: TypeAlias = Array | npt.ArrayLike
_DType: TypeAlias = object | npt.DTypeLike

@type_check_only
class _HasDevice(Protocol[_DeviceT_co]):
    @property
    def device(self, /) -> _DeviceT_co: ...

@type_check_only
class _HasShape(Protocol):
    @property
    def shape(self, /) -> tuple[int, ...]: ...

def _asarray(
    array: ArrayLike,
    dtype: _DType = None,
    order: Literal["K", "A", "C", "F"] | None = None,
    copy: bool | None = None,
    *,
    xp: ModuleType | None = None,
    check_finite: bool = False,
    subok: bool = False,
) -> Array: ...
def array_namespace(*arrays: Array) -> ModuleType: ...
def device(x: _HasDevice[_DeviceT_co], /) -> _DeviceT_co: ...
def size(x: _HasShape) -> int: ...
