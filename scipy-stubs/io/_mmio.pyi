from typing import Any, ClassVar, Literal, TypeAlias, TypedDict, type_check_only
from typing_extensions import Unpack

import numpy as np
import numpy.typing as npt
from scipy._typing import FileLike
from scipy.sparse import coo_matrix, sparray, spmatrix

__all__ = ["MMFile", "mminfo", "mmread", "mmwrite"]

_Format: TypeAlias = Literal["coordinate", "array"]
_Field: TypeAlias = Literal["real", "complex", "pattern", "integer"]
_Symmetry: TypeAlias = Literal["general", "symmetric", "skew-symmetric", "hermitian"]
_Info: TypeAlias = tuple[int, int, int, _Format, _Field, _Symmetry]

@type_check_only
class _MMFileKwargs(TypedDict, total=False):
    rows: int
    cols: int
    entries: int
    format: _Format
    field: _Field
    symmetry: _Symmetry

def asstr(s: object) -> str: ...
def mminfo(source: FileLike[bytes]) -> _Info: ...
def mmread(source: FileLike[bytes]) -> npt.NDArray[np.number[Any]] | coo_matrix: ...
def mmwrite(
    target: FileLike[bytes],
    a: spmatrix | sparray | npt.ArrayLike,
    comment: str = "",
    field: _Field | None = None,
    precision: int | None = None,
    symmetry: _Symmetry | None = None,
) -> None: ...

class MMFile:
    FORMAT_COORDINATE: ClassVar = "coordinate"
    FORMAT_ARRAY: ClassVar = "array"
    FORMAT_VALUES: ClassVar = "coordinate", "array"

    FIELD_INTEGER: ClassVar = "integer"
    FIELD_UNSIGNED: ClassVar = "unsigned-integer"
    FIELD_REAL: ClassVar = "real"
    FIELD_COMPLEX: ClassVar = "complex"
    FIELD_PATTERN: ClassVar = "pattern"
    FIELD_VALUES: ClassVar = "integer", "unsigned-integer", "real", "complex", "pattern"

    SYMMETRY_GENERAL: ClassVar = "general"
    SYMMETRY_SYMMETRIC: ClassVar = "symmetric"
    SYMMETRY_SKEW_SYMMETRIC: ClassVar = "skew-symmetric"
    SYMMETRY_HERMITIAN: ClassVar = "hermitian"
    SYMMETRY_VALUES: ClassVar = "general", "symmetric", "skew-symmetric", "hermitian"

    DTYPES_BY_FIELD: ClassVar[dict[_Field, Literal["intp", "uint64", "d", "D"]]] = ...

    def __init__(self, /, **kwargs: Unpack[_MMFileKwargs]) -> None: ...
    @property
    def rows(self, /) -> int: ...
    @property
    def cols(self, /) -> int: ...
    @property
    def entries(self, /) -> int: ...
    @property
    def format(self, /) -> _Format: ...
    @property
    def field(self, /) -> _Field: ...
    @property
    def symmetry(self, /) -> _Symmetry: ...
    @property
    def has_symmetry(self, /) -> bool: ...
    def read(self, /, source: FileLike[bytes]) -> npt.NDArray[np.number[Any]] | coo_matrix: ...
    def write(
        self,
        /,
        target: FileLike[bytes],
        a: spmatrix | sparray | npt.ArrayLike,
        comment: str = "",
        field: _Field | None = None,
        precision: int | None = None,
        symmetry: _Symmetry | None = None,
    ) -> None: ...
    @classmethod
    def info(cls, /, source: FileLike[bytes]) -> _Info: ...
    @staticmethod
    def reader() -> None: ...
    @staticmethod
    def writer() -> None: ...
