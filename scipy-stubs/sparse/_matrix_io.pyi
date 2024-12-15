from typing import Final, Literal, Protocol, TypeAlias, TypedDict, TypeVar, overload, type_check_only

import optype as op
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._data import _data_matrix
from ._dia import dia_array, dia_matrix

__all__ = ["load_npz", "save_npz"]

_StrOrBytesT = TypeVar("_StrOrBytesT", bound=bytes | str, default=bytes | str)
_StrOrBytesT_co = TypeVar("_StrOrBytesT_co", bound=bytes | str, default=bytes | str, covariant=True)

@type_check_only
class _CanWriteBytes(Protocol):
    def write(self, s: bytes, /) -> object: ...

@type_check_only
class _CanReadAndSeekBytes(Protocol):
    def read(self, length: int = ..., /) -> bytes: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

@type_check_only
class _PickleKwargs(TypedDict):
    allow_pickle: Literal[False]

# A superior version of `os.PathLike` that's actually valid:
# Never parametrize generic types with "a TypeVar with constraints"!
@type_check_only
class _CanFSPath(Protocol[_StrOrBytesT_co]):
    @overload
    def __fspath__(self: _CanFSPath[str], /) -> str: ...
    @overload
    def __fspath__(self: _CanFSPath[bytes], /) -> bytes: ...
    @overload
    def __fspath__(self: _CanFSPath[bytes | str], /) -> bytes | str: ...

_ToPath: TypeAlias = _StrOrBytesT | _CanFSPath[_StrOrBytesT]

_DataArrayOut: TypeAlias = bsr_array | coo_array | csc_array | csr_array | dia_array
_DataMatrixOut: TypeAlias = bsr_matrix | coo_matrix | csc_matrix | csr_matrix | dia_matrix

###

PICKLE_KWARGS: Final[_PickleKwargs] = ...

def load_npz(file: _ToPath | _CanReadAndSeekBytes) -> _DataArrayOut | _DataMatrixOut: ...
def save_npz(file: _ToPath[str] | _CanWriteBytes, matrix: _data_matrix, compressed: op.CanBool = True) -> None: ...
