import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeInt_co

def lambertw(
    z: _ArrayLikeComplex_co,
    k: _ArrayLikeInt_co = 0,
    tol: float | np.float64 = 1e-8,
) -> np.complex128 | npt.NDArray[np.complex128]: ...
