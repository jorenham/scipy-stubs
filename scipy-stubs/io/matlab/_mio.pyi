from collections.abc import Mapping
from typing import Literal, TypeAlias, TypedDict, type_check_only
from typing_extensions import Unpack

import optype.numpy as onp
from scipy._typing import ByteOrder, FileName
from scipy.sparse import coo_array, coo_matrix
from ._miobase import MatFileReader

__all__ = ["loadmat", "savemat", "whosmat"]

_MDict: TypeAlias = Mapping[str, onp.Array2D | coo_array | coo_matrix]
_DataClass: TypeAlias = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "single",
    "double",
    "cell",
    "struct",
    "object",
    "char",
    "sparse",
    "function",
    "opaque",
    "logical",
    "unknown",
]

@type_check_only
class _ReaderKwargs(TypedDict, total=False):
    byte_order: ByteOrder | None
    mat_dtype: bool
    squeeze_me: bool
    chars_as_strings: bool
    matlab_compatible: bool
    struct_as_record: bool
    verify_compressed_data_integrity: bool
    simplify_cells: bool
    variable_names: list[str] | tuple[str] | None

###

def mat_reader_factory(
    file_name: FileName,
    appendmat: bool = True,
    **kwargs: Unpack[_ReaderKwargs],
) -> tuple[MatFileReader, bool]: ...

#
def loadmat(
    file_name: FileName,
    mdict: _MDict | None = None,
    appendmat: bool = True,
    *,
    spmatrix: bool = True,
    **kwargs: Unpack[_ReaderKwargs],
) -> _MDict: ...

#
def savemat(
    file_name: FileName,
    mdict: _MDict,
    appendmat: bool = True,
    format: Literal["5", "4"] = "5",
    long_field_names: bool = False,
    do_compression: bool = False,
    oned_as: Literal["row", "column"] = "row",
) -> None: ...

#
def whosmat(
    file_name: FileName,
    appendmat: bool = True,
    **kwargs: Unpack[_ReaderKwargs],
) -> list[tuple[str, tuple[int, ...], _DataClass]]: ...
