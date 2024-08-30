import array as _array
import mmap
from typing import Any, Literal, Protocol, TypeAlias, TypeVar
from typing_extensions import Buffer

import numpy as np
import optype as op

__all__ = ["NestedSequence", "SupportsBufferProtocol"]

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...

SupportsBufferProtocol: TypeAlias = (
    Buffer
    | bytes
    | bytearray
    | memoryview
    | _array.array[Any]
    | mmap.mmap
    | np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    | np.generic
)
Array: TypeAlias = op.CanGetitem[op.CanIndex | slice | tuple[op.CanIndex | slice, ...], Array | complex]
Device: TypeAlias = Literal["CPU"] | object
