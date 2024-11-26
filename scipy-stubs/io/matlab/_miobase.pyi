import abc
from collections.abc import Mapping
from typing import IO, Final, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from scipy._typing import ByteOrder, FileName

__all__ = ["MatReadError", "MatReadWarning", "MatWriteError"]

_T = TypeVar("_T", bound=op.HasDoc)

class MatReadError(Exception): ...
class MatWriteError(Exception): ...
class MatReadWarning(UserWarning): ...

class MatVarReader:
    def __init__(self, /, file_reader: MatFileReader) -> None: ...
    @abc.abstractmethod
    def read_header(self, /) -> dict[str, object]: ...
    @abc.abstractmethod
    def array_from_header(self, /, header: Mapping[str, object]) -> onp.ArrayND: ...

class MatFileReader:
    mat_stream: Final[IO[bytes]]
    dtypes: Final[Mapping[str, np.dtype[np.generic]]]
    byte_order: Final[ByteOrder]
    struct_as_record: Final[bool]
    verify_compressed_data_integrity: Final[bool]
    simplify_cells: Final[bool]
    mat_dtype: bool
    squeeze_me: bool
    chars_as_strings: bool

    def __init__(
        self,
        /,
        mat_stream: IO[bytes],
        byte_order: ByteOrder | None = None,
        mat_dtype: bool = False,
        squeeze_me: bool = False,
        chars_as_strings: bool = True,
        matlab_compatible: bool = False,
        struct_as_record: bool = True,
        verify_compressed_data_integrity: bool = True,
        simplify_cells: bool = False,
    ) -> None: ...
    def set_matlab_compatible(self, /) -> None: ...
    def guess_byte_order(self, /) -> ByteOrder: ...
    def end_of_stream(self, /) -> bool: ...

doc_dict: Final[dict[str, str]] = ...
get_matfile_version = matfile_version

def _get_matfile_version(fileobj: IO[bytes]) -> tuple[Literal[1, 2], int]: ...
def docfiller(f: _T) -> _T: ...
def convert_dtypes(dtype_template: Mapping[str, str], order_code: ByteOrder) -> dict[str, np.dtype[np.generic]]: ...
def read_dtype(mat_stream: IO[bytes], a_dtype: npt.DTypeLike) -> onp.ArrayND: ...
def matfile_version(file_name: FileName, *, appendmat: bool = True) -> tuple[Literal[0, 1, 2], int]: ...
def matdims(arr: onp.ArrayND, oned_as: Literal["column", "row"] = "column") -> onp.AtLeast1D: ...
def arr_dtype_number(arr: onp.HasDType[np.dtype[np.character]], num: int | str) -> np.dtype[np.character]: ...
def arr_to_chars(arr: onp.ArrayND[np.str_]) -> onp.ArrayND[np.str_]: ...
