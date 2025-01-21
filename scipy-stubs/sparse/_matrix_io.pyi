from typing import Final, Literal, Protocol, TypeAlias, TypedDict, type_check_only

import optype as op
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._data import _data_matrix
from ._dia import dia_array, dia_matrix

__all__ = ["load_npz", "save_npz"]

_DataArrayOut: TypeAlias = bsr_array | coo_array | csc_array | csr_array | dia_array
_DataMatrixOut: TypeAlias = bsr_matrix | coo_matrix | csc_matrix | csr_matrix | dia_matrix

@type_check_only
class _CanReadAndSeekBytes(Protocol):
    def read(self, length: int = ..., /) -> bytes: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

@type_check_only
class _PickleKwargs(TypedDict):
    allow_pickle: Literal[False]

###

PICKLE_KWARGS: Final[_PickleKwargs] = ...

def load_npz(file: op.io.ToPath | _CanReadAndSeekBytes) -> _DataArrayOut | _DataMatrixOut: ...
def save_npz(file: op.io.ToPath[str] | op.io.CanWrite[bytes], matrix: _data_matrix, compressed: op.CanBool = True) -> None: ...
