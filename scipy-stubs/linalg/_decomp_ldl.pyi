from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["ldl"]

_Array_i_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

def ldl(
    A: npt.ArrayLike,
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
