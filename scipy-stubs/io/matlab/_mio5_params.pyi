from typing import Literal
from typing_extensions import Self, override

import numpy as np
from scipy._typing import Untyped, UntypedDict

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

miINT8: int
miUINT8: int
miINT16: int
miUINT16: int
miINT32: int
miUINT32: int
miSINGLE: int
miDOUBLE: int
miINT64: int
miUINT64: int
miMATRIX: int
miCOMPRESSED: int
miUTF8: int
miUTF16: int
miUTF32: int
mxCELL_CLASS: int
mxSTRUCT_CLASS: int
mxOBJECT_CLASS: int
mxCHAR_CLASS: int
mxSPARSE_CLASS: int
mxDOUBLE_CLASS: int
mxSINGLE_CLASS: int
mxINT8_CLASS: int
mxUINT8_CLASS: int
mxINT16_CLASS: int
mxUINT16_CLASS: int
mxINT32_CLASS: int
mxUINT32_CLASS: int
mxINT64_CLASS: int
mxUINT64_CLASS: int
mxFUNCTION_CLASS: int
mxOPAQUE_CLASS: int
mxOBJECT_CLASS_FROM_MATRIX_H: int
mdtypes_template: UntypedDict
mclass_dtypes_template: dict[int, str]
mclass_info: dict[int, str]
NP_TO_MTYPES: dict[str, int]
NP_TO_MXTYPES: dict[str, int]
codecs_template: dict[int, dict[str, str]]
MDTYPES: UntypedDict

class mat_struct: ...

class MatlabObject(np.ndarray[tuple[int, ...], np.dtype[np.void]]):
    classname: str | None

    def __new__(cls, input_array: np.ndarray[tuple[int, ...], np.dtype[np.void]], classname: Untyped | None = None) -> Self: ...
    @override
    def __array_finalize__(self, obj: Self) -> None: ...  # type: ignore[override]

class MatlabFunction(np.ndarray[tuple[int, ...], np.dtype[np.void]]):
    def __new__(cls, input_array: np.ndarray[tuple[int, ...], np.dtype[np.void]]) -> Self: ...

class MatlabOpaque(np.ndarray[tuple[int, ...], np.dtype[np.void]]):
    def __new__(cls, input_array: np.ndarray[tuple[int, ...], np.dtype[np.void]]) -> Self: ...

OPAQUE_DTYPE: np.dtypes.VoidDType[Literal[32]]
