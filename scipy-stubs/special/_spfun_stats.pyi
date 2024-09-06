import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
import scipy._typing as spt

__all__ = ["multigammaln"]

def multigammaln(a: _ArrayLikeFloat_co, d: spt.AnyInt) -> np.float64 | npt.NDArray[np.float64]: ...
