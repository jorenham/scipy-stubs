from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["polar"]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

def polar(a: npt.ArrayLike, side: Literal["left", "right"] = "right") -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
