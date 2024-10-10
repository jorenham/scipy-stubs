from collections.abc import Sequence
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _SupportsArray

__all__ = (
    "_BoolValueIn",
    "_ComplexArrayIn",
    "_ComplexArrayOut",
    "_ComplexArrayOutT",
    "_ComplexValueIn",
    "_ComplexValueOut",
    "_FloatArrayIn",
    "_FloatArrayOut",
    "_FloatArrayOutT",
    "_FloatMatrixIn",
    "_FloatValueIn",
    "_FloatValueOut",
    "_FloatVectorIn",
    "_IntValueIn",
    "_NumberArrayOutT",
)

_BoolValueIn: TypeAlias = bool | np.bool_ | Literal[0, 1]
_IntValueIn: TypeAlias = int | np.integer[Any]

_FloatValueIn: TypeAlias = float | np.floating[Any] | np.integer[Any]
_FloatVectorIn: TypeAlias = _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]] | Sequence[_FloatValueIn]
_FloatMatrixIn: TypeAlias = _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]] | Sequence[_FloatVectorIn]
_FloatArrayIn: TypeAlias = _SupportsArray[np.dtype[np.floating[Any] | np.integer[Any]]] | Sequence[float | _FloatArrayIn]
_FloatValueOut: TypeAlias = np.float64 | np.float32
_FloatArrayOut: TypeAlias = npt.NDArray[_FloatValueOut]
_FloatArrayOutT = TypeVar("_FloatArrayOutT", bound=_FloatArrayOut, default=_FloatArrayOut)

_ComplexValueIn: TypeAlias = complex | np.number[Any]
_ComplexArrayIn: TypeAlias = _SupportsArray[np.dtype[np.number[Any]]] | Sequence[complex | _ComplexArrayIn]
_ComplexValueOut: TypeAlias = _FloatValueOut | np.complex128 | np.complex64
_ComplexArrayOut: TypeAlias = npt.NDArray[_ComplexValueOut]
_ComplexArrayOutT = TypeVar("_ComplexArrayOutT", bound=_ComplexArrayOut, default=_ComplexArrayOut)

_NumberArrayOutT = TypeVar("_NumberArrayOutT", bound=npt.NDArray[np.number[Any]], default=npt.NDArray[np.number[Any]])
