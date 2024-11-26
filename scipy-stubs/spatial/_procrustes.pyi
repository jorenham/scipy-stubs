import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["procrustes"]

def procrustes(
    data1: npt.ArrayLike,
    data2: npt.ArrayLike,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], np.float64]: ...
