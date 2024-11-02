from typing import IO, Final, Literal, TypeAlias, type_check_only
from typing_extensions import LiteralString, Protocol, Self

import optype.typing as opt
from scipy._typing import FileName
from scipy.sparse import csc_matrix, sparray, spmatrix

__all__ = ["hb_read", "hb_write"]

_ValueType: TypeAlias = Literal["real", "complex", "pattern", "integer"]
_Structure: TypeAlias = Literal["symmetric", "unsymmetric", "hermitian", "skewsymmetric", "rectangular"]
_Storage: TypeAlias = Literal["assembled", "elemental"]

@type_check_only
class _HasWidthAndRepeat(Protocol):
    @property
    def width(self, /) -> int: ...
    @property
    def repeat(self, /) -> int | None: ...

###

class MalformedHeader(Exception): ...
class LineOverflow(Warning): ...

class HBInfo:
    title: Final[str]
    key: Final[str]
    total_nlines: Final[int]
    pointer_nlines: Final[int]
    indices_nlines: Final[int]
    values_nlines: Final[int]
    pointer_format: Final[int]
    indices_format: Final[int]
    values_format: Final[int]
    pointer_dtype: Final[int]
    indices_dtype: Final[int]
    values_dtype: Final[int]
    pointer_nbytes_full: Final[int]
    indices_nbytes_full: Final[int]
    values_nbytes_full: Final[int]
    nrows: Final[int]
    ncols: Final[int]
    nnon_zeros: Final[int]
    nelementals: Final[int]
    mxtype: HBMatrixType

    @classmethod
    def from_data(
        cls,
        m: sparray | spmatrix,
        title: str = "Default title",
        key: str = "0",
        mxtype: HBMatrixType | None = None,
        fmt: None = None,
    ) -> Self: ...
    @classmethod
    def from_file(cls, fid: IO[str]) -> Self: ...
    def __init__(
        self,
        /,
        title: str,
        key: str,
        total_nlines: int,
        pointer_nlines: int,
        indices_nlines: int,
        values_nlines: int,
        mxtype: HBMatrixType,
        nrows: int,
        ncols: int,
        nnon_zeros: int,
        pointer_format_str: str,
        indices_format_str: str,
        values_format_str: str,
        right_hand_sides_nlines: int = 0,
        nelementals: int = 0,
    ) -> None: ...
    def dump(self, /) -> str: ...

class HBMatrixType:
    value_type: Final[_ValueType]
    structure: Final[_Structure]
    storage: Final[_Storage]
    @property
    def fortran_format(self, /) -> LiteralString: ...
    @classmethod
    def from_fortran(cls, fmt: str) -> Self: ...
    def __init__(self, /, value_type: _ValueType, structure: _Structure, storage: _Storage = "assembled") -> None: ...

class HBFile:
    def __init__(self, /, file: IO[str], hb_info: HBMatrixType | None = None) -> None: ...
    @property
    def title(self, /) -> str: ...
    @property
    def key(self, /) -> str: ...
    @property
    def type(self, /) -> _ValueType: ...
    @property
    def structure(self, /) -> _Structure: ...
    @property
    def storage(self, /) -> _Storage: ...
    def read_matrix(self, /) -> csc_matrix: ...
    def write_matrix(self, /, m: spmatrix | sparray) -> None: ...

def _nbytes_full(fmt: _HasWidthAndRepeat, nlines: int) -> int: ...
def _expect_int(value: opt.AnyInt, msg: str | None = None) -> int: ...
def _read_hb_data(content: IO[str], header: HBInfo) -> csc_matrix: ...
def _write_data(m: sparray | spmatrix, fid: IO[str], header: HBInfo) -> None: ...
def hb_read(path_or_open_file: IO[str] | FileName) -> csc_matrix: ...
def hb_write(path_or_open_file: IO[str] | FileName, m: spmatrix | sparray, hb_info: HBInfo | None = None) -> None: ...
