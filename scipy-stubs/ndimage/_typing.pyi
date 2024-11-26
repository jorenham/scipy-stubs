from typing import Any, TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = (
    "_ComplexArrayOut",
    "_ComplexValueOut",
    "_FloatArrayOut",
    "_FloatValueOut",
    "_ScalarArrayOut",
    "_ScalarValueOut",
)

_FloatValueOut: TypeAlias = np.float64 | np.float32
_FloatArrayOut: TypeAlias = onp.ArrayND[_FloatValueOut]

_ComplexValueOut: TypeAlias = _FloatValueOut | np.complex128 | np.complex64
_ComplexArrayOut: TypeAlias = onp.ArrayND[_ComplexValueOut]

_ScalarValueOut: TypeAlias = np.number[Any] | np.bool_
_ScalarArrayOut: TypeAlias = onp.ArrayND[_ScalarValueOut]
