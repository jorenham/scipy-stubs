import io
from typing import Any, Final, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypedDict, Unpack, override

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, FileLike, FileName, Truthy
from scipy.sparse import coo_array, coo_matrix
from scipy.sparse._base import _spbase

__all__ = ["mminfo", "mmread", "mmwrite"]

_Format: TypeAlias = Literal["coordinate", "array"]
_Field: TypeAlias = Literal["real", "complex", "pattern", "integer"]
_Symmetry: TypeAlias = Literal["general", "symmetric", "skew-symmetric", "hermitian"]

@type_check_only
class _TextToBytesWrapperKwargs(TypedDict, total=False):
    buffer_size: int

###

PARALLELISM: Final = 0
ALWAYS_FIND_SYMMETRY: Final = False

class _TextToBytesWrapper(io.BufferedReader):
    encoding: Final[str]
    errors: Final[str]

    def __init__(
        self,
        /,
        text_io_buffer: io.TextIOBase,
        encoding: str | None = None,
        errors: str | None = None,
        **kwargs: Unpack[_TextToBytesWrapperKwargs],
    ) -> None: ...
    @override
    def read(self, /, size: int | None = -1) -> bytes: ...
    @override
    def read1(self, /, size: int = -1) -> bytes: ...
    @override
    def peek(self, /, size: int = -1) -> bytes: ...
    @override
    def seek(self, /, offset: int, whence: int = 0) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

@overload
def mmread(source: FileLike[bytes], *, spmatrix: Truthy = True) -> onp.ArrayND[np.number[Any]] | coo_array: ...
@overload
def mmread(source: FileLike[bytes], *, spmatrix: Falsy) -> onp.ArrayND[np.number[Any]] | coo_matrix: ...

#
def mmwrite(
    target: FileName,
    a: onp.CanArray | list[object] | tuple[object, ...] | _spbase,
    comment: str | None = None,
    field: _Field | None = None,
    precision: int | None = None,
    symmetry: _Symmetry | Literal["AUTO"] = "AUTO",
) -> None: ...

#
def mminfo(source: FileName) -> tuple[int, int, int, _Format, _Field, _Symmetry]: ...
