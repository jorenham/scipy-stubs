from typing import Final, Literal, TypeAlias, final
from typing_extensions import Self, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = [
    "MDTYPES",
    "NP_TO_MTYPES",
    "NP_TO_MXTYPES",
    "OPAQUE_DTYPE",
    "MatlabFunction",
    "MatlabObject",
    "MatlabOpaque",
    "codecs_template",
    "mat_struct",
    "mclass_dtypes_template",
    "mclass_info",
    "mdtypes_template",
    "miCOMPRESSED",
    "miDOUBLE",
    "miINT8",
    "miINT16",
    "miINT32",
    "miINT64",
    "miMATRIX",
    "miSINGLE",
    "miUINT8",
    "miUINT16",
    "miUINT32",
    "miUINT64",
    "miUTF8",
    "miUTF16",
    "miUTF32",
    "mxCELL_CLASS",
    "mxCHAR_CLASS",
    "mxDOUBLE_CLASS",
    "mxFUNCTION_CLASS",
    "mxINT8_CLASS",
    "mxINT16_CLASS",
    "mxINT32_CLASS",
    "mxINT64_CLASS",
    "mxOBJECT_CLASS",
    "mxOBJECT_CLASS_FROM_MATRIX_H",
    "mxOPAQUE_CLASS",
    "mxSINGLE_CLASS",
    "mxSPARSE_CLASS",
    "mxSTRUCT_CLASS",
    "mxUINT8_CLASS",
    "mxUINT16_CLASS",
    "mxUINT32_CLASS",
    "mxUINT64_CLASS",
]

_MType: TypeAlias = Literal[1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18]
_MXType: TypeAlias = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

miINT8: Final = 1
miUINT8: Final = 2
miINT16: Final = 3
miUINT16: Final = 4
miINT32: Final = 5
miUINT32: Final = 6
miSINGLE: Final = 7
miDOUBLE: Final = 9
miINT64: Final = 12
miUINT64: Final = 13
miMATRIX: Final = 14
miCOMPRESSED: Final = 15
miUTF8: Final = 16
miUTF16: Final = 17
miUTF32: Final = 18

mxCELL_CLASS: Final = 1
mxSTRUCT_CLASS: Final = 2
mxOBJECT_CLASS: Final = 3
mxCHAR_CLASS: Final = 4
mxSPARSE_CLASS: Final = 5
mxDOUBLE_CLASS: Final = 6
mxSINGLE_CLASS: Final = 7
mxINT8_CLASS: Final = 8
mxUINT8_CLASS: Final = 9
mxINT16_CLASS: Final = 10
mxUINT16_CLASS: Final = 11
mxINT32_CLASS: Final = 12
mxUINT32_CLASS: Final = 13
mxINT64_CLASS: Final = 14
mxUINT64_CLASS: Final = 15
mxFUNCTION_CLASS: Final = 16
mxOPAQUE_CLASS: Final = 17
mxOBJECT_CLASS_FROM_MATRIX_H: Final = 18

codecs_template: dict[Literal[16, 17, 18], dict[Literal["codec", "width"], Literal["utf_8", "utf_16", "utf_32", 1, 2, 4]]]
mdtypes_template: Final[dict[_MType | str, str | list[tuple[str, str]]]]
mclass_info: Final[dict[_MXType, str]]
mclass_dtypes_template: Final[dict[_MXType, str]]

NP_TO_MTYPES: Final[dict[str, _MType]]
NP_TO_MXTYPES: Final[dict[str, _MXType]]
MDTYPES: Final[dict[Literal["<", ">"], dict[str, dict[_MType, str]]]]
OPAQUE_DTYPE: Final[np.dtypes.VoidDType[Literal[32]]]

@final
class mat_struct: ...

class MatlabObject(np.ndarray[tuple[int, ...], np.dtypes.VoidDType[int]]):
    classname: Final[str | None]

    def __new__(cls, input_array: onpt.AnyVoidArray, classname: str | None = None) -> Self: ...
    @override
    def __array_finalize__(self, /, obj: None | npt.NDArray[np.void]) -> None: ...

class MatlabFunction(np.ndarray[tuple[int, ...], np.dtype[np.void]]):
    @override
    def __new__(cls, input_array: onpt.AnyVoidArray) -> Self: ...

class MatlabOpaque(np.ndarray[tuple[int, ...], np.dtype[np.void]]):
    @override
    def __new__(cls, input_array: onpt.AnyVoidArray) -> Self: ...
