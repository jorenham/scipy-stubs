from types import ModuleType
from typing import Any, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import AnyBool, OrderKACF

__all__ = ["_asarray", "array_namespace", "device", "size"]

_0: TypeAlias = Literal[0]  # noqa: PYI042
_1: TypeAlias = Literal[1]  # noqa: PYI042

# TODO(jorenham): Narrow this down (even though the array-api forgot to specify the type of the a dtype...)
_DType: TypeAlias = type | str | np.dtype[np.generic]
_Device: TypeAlias = Any

_SizeT = TypeVar("_SizeT", bound=int)
_ShapeT_co = TypeVar("_ShapeT_co", covariant=True, bound=tuple[int, ...], default=tuple[int, ...])

_DTypeT_co = TypeVar("_DTypeT_co", bound=_DType, covariant=True, default=_DType)

_DeviceT = TypeVar("_DeviceT")
_DeviceT_co = TypeVar("_DeviceT_co", covariant=True, default=_Device)

@type_check_only
class _HasShape(Protocol[_ShapeT_co]):
    @property
    def shape(self, /) -> _ShapeT_co: ...

@type_check_only
class _HasDevice(Protocol[_DeviceT_co]):
    @property
    def device(self, /) -> _DeviceT_co: ...

# TODO(jorenham): Implement this properly in `optype`:
# https://github.com/jorenham/optype/issues/25
@type_check_only
class _HasArrayAttrs(_HasShape[_ShapeT_co], _HasDevice[_DeviceT_co], Protocol[_ShapeT_co, _DTypeT_co, _DeviceT_co]):
    @property
    def dtype(self, /) -> _DTypeT_co: ...
    @property
    def ndim(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...

    # TODO(jorenham): Use HKT for `T` and `mT` once implemented (the community has been asking for 6 years, but to no avail)
    # https://github.com/python/typing/issues/548
    @property
    def T(self, /) -> _HasArrayAttrs[tuple[int, ...], _DTypeT_co, _DeviceT_co]: ...
    @property
    def mT(self, /) -> _HasArrayAttrs[tuple[int, ...], _DTypeT_co, _DeviceT_co]: ...

###

Array: TypeAlias = _HasArrayAttrs[_ShapeT_co, _DTypeT_co, _DeviceT_co]
ArrayLike: TypeAlias = Array | onp.ToFloatND

def _asarray(
    array: ArrayLike,
    dtype: _DType | None = None,
    order: OrderKACF | None = None,
    copy: AnyBool | None = None,
    *,
    xp: ModuleType | None = None,
    check_finite: AnyBool = False,
    subok: AnyBool = False,
) -> Array: ...
def array_namespace(*arrays: Array) -> ModuleType: ...
def device(x: _HasDevice[_DeviceT], /) -> _DeviceT: ...
@overload
def size(x: _HasShape[tuple[()] | tuple[_0, ...]]) -> _0: ...
@overload
def size(x: _HasShape[tuple[_SizeT] | onp.AtLeast1D[_SizeT, _1] | tuple[_1, _SizeT]]) -> _SizeT: ...
@overload
def size(x: _HasShape) -> int: ...
