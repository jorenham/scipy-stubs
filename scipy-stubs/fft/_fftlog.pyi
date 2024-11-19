from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co

__all__ = ["fht", "fhtoffset", "ifht"]

_ArrayReal: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]

def fht(
    a: _ArrayLikeFloat_co,
    dln: onp.ToFloat,
    mu: onp.ToFloat,
    offset: onp.ToFloat = 0.0,
    bias: onp.ToFloat = 0.0,
) -> _ArrayReal: ...
def ifht(
    A: _ArrayLikeFloat_co,
    dln: onp.ToFloat,
    mu: onp.ToFloat,
    offset: onp.ToFloat = 0.0,
    bias: onp.ToFloat = 0.0,
) -> _ArrayReal: ...
def fhtoffset(dln: onp.ToFloat, mu: onp.ToFloat, initial: onp.ToFloat = 0.0, bias: onp.ToFloat = 0.0) -> np.float64: ...
