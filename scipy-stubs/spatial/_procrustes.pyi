import numpy as np
import numpy.typing as npt

__all__ = ["procrustes"]

def procrustes(
    data1: npt.ArrayLike,
    data2: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64]: ...
