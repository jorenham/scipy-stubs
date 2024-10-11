from collections.abc import Sequence
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _SupportsArray

__all__ = (
    "_BoolValueIn",
    "_ComplexArrayOut",
    "_ComplexArrayOutT",
    "_ComplexValueOut",
    "_FloatArrayIn",
    "_FloatArrayOut",
    "_FloatArrayOutT",
    "_FloatMatrixIn",
    "_FloatValueIn",
    "_FloatValueOut",
    "_FloatVectorIn",
    "_IntArrayIn",
    "_IntValueIn",
    "_ScalarArrayIn",
    "_ScalarArrayOut",
    "_ScalarArrayOutT",
    "_ScalarValueIn",
    "_ScalarValueIn",
    "_ScalarValueOut",
)

_BoolValueIn: TypeAlias = bool | Literal[0, 1] | np.bool_

__IntValue: TypeAlias = np.integer[Any] | np.bool_
_IntValueIn: TypeAlias = int | __IntValue
_IntArrayIn: TypeAlias = int | _SupportsArray[np.dtype[__IntValue]] | Sequence[_IntArrayIn]

__FloatValue: TypeAlias = np.floating[Any] | __IntValue
_FloatValueIn: TypeAlias = float | __FloatValue
_FloatVectorIn: TypeAlias = float | Sequence[_FloatValueIn] | _SupportsArray[np.dtype[__FloatValue]]
_FloatMatrixIn: TypeAlias = _FloatVectorIn | Sequence[_FloatVectorIn]
_FloatArrayIn: TypeAlias = float | _SupportsArray[np.dtype[__FloatValue]] | Sequence[_FloatArrayIn]

_FloatValueOut: TypeAlias = np.float64 | np.float32
_FloatArrayOut: TypeAlias = npt.NDArray[_FloatValueOut]
_FloatArrayOutT = TypeVar("_FloatArrayOutT", bound=_FloatArrayOut, default=_FloatArrayOut)

_ComplexValueOut: TypeAlias = _FloatValueOut | np.complex128 | np.complex64
_ComplexArrayOut: TypeAlias = npt.NDArray[_ComplexValueOut]
_ComplexArrayOutT = TypeVar("_ComplexArrayOutT", bound=_ComplexArrayOut, default=_ComplexArrayOut)

_ScalarValueIn: TypeAlias = complex | np.number[Any] | np.bool_
_ScalarArrayIn: TypeAlias = complex | _SupportsArray[np.dtype[np.number[Any]]] | Sequence[_ScalarArrayIn]

_ScalarValueOut: TypeAlias = np.number[Any] | np.bool_
_ScalarArrayOut: TypeAlias = npt.NDArray[_ScalarValueOut]
_ScalarArrayOutT = TypeVar("_ScalarArrayOutT", bound=_ScalarArrayOut, default=_ScalarArrayOut)
