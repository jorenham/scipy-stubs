from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal

__all__ = ["fht", "fhtoffset", "ifht"]

_ArrayReal: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]

def fht(a: _ArrayLikeFloat_co, dln: AnyReal, mu: AnyReal, offset: AnyReal = 0.0, bias: AnyReal = 0.0) -> _ArrayReal: ...
def ifht(A: _ArrayLikeFloat_co, dln: AnyReal, mu: AnyReal, offset: AnyReal = 0.0, bias: AnyReal = 0.0) -> _ArrayReal: ...
def fhtoffset(dln: AnyReal, mu: AnyReal, initial: AnyReal = 0.0, bias: AnyReal = 0.0) -> np.float64: ...
