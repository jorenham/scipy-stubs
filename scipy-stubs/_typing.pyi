# Helper types for internal use (type-check only).

from typing import TypeAlias

import numpy as np

__all__ = ("ArrayLike0D", "Untyped")

Untyped: TypeAlias = object

# keep in sync with `numpy.typing._scalar_like._ScalarLike`
ArrayLike0D: TypeAlias = bool | int | float | complex | str | bytes | np.generic
