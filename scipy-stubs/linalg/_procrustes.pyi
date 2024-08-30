from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["orthogonal_procrustes"]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

def orthogonal_procrustes(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, float | np.float64]: ...
