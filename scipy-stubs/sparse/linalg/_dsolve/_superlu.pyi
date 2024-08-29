from typing import Any

import numpy as np

import numpy.typing as npt
import optype.numpy as onpt

from scipy.sparse import csc_matrix
from scipy.stats._levy_stable import np

__all__ = ["SuperLU"]

class SuperLU:
    L: csc_matrix
    U: csc_matrix
    nnz: int
    perm_r: onpt.Array[tuple[int], np.intp]
    perm_c: onpt.Array[tuple[int], np.intp]
    shape: tuple[int, ...]

    def solve(self, /, rhs: npt.NDArray[np.number[Any]]) -> npt.NDArray[np.number[Any]]: ...
