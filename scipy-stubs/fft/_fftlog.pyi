from typing import TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = ["fht", "fhtoffset", "ifht"]

_RealND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]

def fht(
    a: onp.ToFloatND,
    dln: onp.ToFloat,
    mu: onp.ToFloat,
    offset: onp.ToFloat = 0.0,
    bias: onp.ToFloat = 0.0,
) -> _RealND: ...
def ifht(
    A: onp.ToFloatND,
    dln: onp.ToFloat,
    mu: onp.ToFloat,
    offset: onp.ToFloat = 0.0,
    bias: onp.ToFloat = 0.0,
) -> _RealND: ...
def fhtoffset(dln: onp.ToFloat, mu: onp.ToFloat, initial: onp.ToFloat = 0.0, bias: onp.ToFloat = 0.0) -> np.float64: ...
