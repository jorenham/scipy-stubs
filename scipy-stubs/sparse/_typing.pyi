# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeAliasType

import numpy as np
import optype as op

__all__ = "Complex", "Float", "Int", "SPFormat", "Scalar", "ToShape", "ToShape1D", "ToShape2D"

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

###

SPFormat: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

###

ToShape1D: TypeAlias = tuple[op.CanIndex]
ToShape2D: TypeAlias = tuple[op.CanIndex, op.CanIndex]
ToShape: TypeAlias = ToShape1D | ToShape2D
