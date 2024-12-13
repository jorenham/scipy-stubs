# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeAliasType, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt

__all__ = (
    "Complex",
    "Float",
    "Index1D",
    "Int",
    "Matrix",
    "SPFormat",
    "Scalar",
    "ToDType",
    "ToDTypeBool",
    "ToDTypeComplex",
    "ToDTypeFloat",
    "ToDTypeInt",
    "ToShape",
    "ToShape1D",
    "ToShape2D",
)

###

# NOTE: This is roughly speaking equivalent to `[u]int8 | [u]int16 | [u]int32 | [u]int64` (on most modern platforms)
Int: TypeAlias = np.integer[Any]
# NOTE: NOT equivalent to `floating`! It's considered invalid to use `float16` in `scipy.sparse`.
Float: TypeAlias = np.float32 | np.float64 | np.longdouble
# NOTE: This is (almost always) equivalent to `complex64 | complex128 | clongdouble`.
Complex: TypeAlias = np.complexfloating[Any, Any]
# NOTE: Roughly speaking, this is equivalent to `number[Any] - float16` (the `-` denotes the set difference analogue for types)
# NOTE: This (almost always) matches `._sputils.supported_dtypes`
# NOTE: The `TypeAliasType` is used to avoid long error messages.
Scalar = TypeAliasType("Scalar", np.bool_ | Int | Float | Complex)

Matrix: TypeAlias = np.matrix[tuple[int, int], np.dtype[_SCT]]
Index1D: TypeAlias = onp.Array1D[np.int32 | np.int64]

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

_SCT = TypeVar("_SCT", bound=np.generic)
ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
ToDTypeBool: TypeAlias = type[bool] | ToDType[np.bool_] | Literal["bool", "bool_", "?", "b1"]
ToDTypeInt: TypeAlias = type[opt.JustInt] | ToDType[np.int_] | Literal["int", "int_", "n"]
ToDTypeFloat: TypeAlias = type[opt.Just[float]] | ToDType[np.float64] | Literal["float", "float64", "double", "f8", "d"]
ToDTypeComplex: TypeAlias = (
    type[opt.Just[complex]]
    | ToDType[np.complex128]
    | Literal["complex", "complex128", "cdouble", "c16", "D"]
)  # fmt: skip

ToShape1D: TypeAlias = tuple[op.CanIndex]
ToShape2D: TypeAlias = tuple[op.CanIndex, op.CanIndex]
ToShape: TypeAlias = ToShape1D | ToShape2D
