# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeAliasType, TypeVar, Unpack

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt

__all__ = (
    "Complex",
    "Float",
    "Index1D",
    "Int",
    "SPFormat",
    "Scalar",
    "Shape",
    "ShapeBSR",
    "ShapeCOO",
    "ShapeCSC",
    "ShapeCSR",
    "ShapeDIA",
    "ShapeDOK",
    "ToDType",
    "ToDTypeBool",
    "ToDTypeComplex",
    "ToDTypeFloat",
    "ToDTypeInt",
    "ToShape1d",
    "ToShape1d",
    "ToShape1d2d",
    "ToShape1dNd",
    "ToShape2d",
    "ToShape2dNd",
    "ToShape3dNd",
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

Index1D: TypeAlias = onp.Array1D[np.int32 | np.int64]

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

_SCT = TypeVar("_SCT", bound=np.generic)
ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]
ToDTypeBool: TypeAlias = onp.AnyBoolDType
ToDTypeInt: TypeAlias = type[opt.JustInt] | onp.AnyIntDType  # see https://github.com/jorenham/optype/issues/235
ToDTypeFloat: TypeAlias = onp.AnyFloat64DType
ToDTypeComplex: TypeAlias = onp.AnyComplex128DType

ToShape1d: TypeAlias = tuple[op.CanIndex]  # ndim == 1
ToShape2d: TypeAlias = tuple[op.CanIndex, op.CanIndex]  # ndim == 2
ToShape1d2d: TypeAlias = ToShape2d | ToShape1d  # 1 <= ndim <= 2
ToShape1dNd: TypeAlias = tuple[op.CanIndex, Unpack[tuple[op.CanIndex, ...]]]  # ndim >= 1
ToShape2dNd: TypeAlias = tuple[op.CanIndex, op.CanIndex, Unpack[tuple[op.CanIndex, ...]]]  # ndim >= 2
ToShape3dNd: TypeAlias = tuple[op.CanIndex, op.CanIndex, op.CanIndex, Unpack[tuple[op.CanIndex, ...]]]  # ndim >= 2

Shape: TypeAlias = onp.AtLeast1D
ShapeBSR: TypeAlias = tuple[int, int]
ShapeCOO: TypeAlias = onp.AtLeast1D
ShapeCSC: TypeAlias = tuple[int, int]
ShapeCSR: TypeAlias = tuple[int, int] | tuple[int]
ShapeDIA: TypeAlias = tuple[int, int]
ShapeDOK: TypeAlias = tuple[int, int] | tuple[int]
