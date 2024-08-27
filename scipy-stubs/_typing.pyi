# Helper types for internal use (type-check only).
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

__all__ = "AnyBool", "AnyChar", "AnyComplex", "AnyInt", "AnyReal", "AnyScalar", "Array0D", "Untyped"

# placeholder for missing annotations
Untyped: TypeAlias = object

_SCT = TypeVar("_SCT", bound=np.generic, default=np.generic)
Array0D: TypeAlias = np.ndarray[tuple[()], np.dtype[_SCT]]

# keep in sync with `numpy._typing._scalars`
AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]
AnyInt: TypeAlias = int | np.integer[npt.NBitBase] | np.bool_
AnyReal: TypeAlias = int | float | np.floating[npt.NBitBase] | np.integer[npt.NBitBase] | np.bool_
AnyComplex: TypeAlias = int | float | complex | np.number[npt.NBitBase] | np.bool_
AnyChar: TypeAlias = str | bytes  # `np.str_ <: builtins.str` and `np.bytes_ <: builtins.bytes`
AnyScalar: TypeAlias = int | float | complex | AnyChar | np.generic
