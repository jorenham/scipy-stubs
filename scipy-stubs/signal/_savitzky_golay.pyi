from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped, UntypedArray

_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]

def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: Literal["conv", "dot"] = "conv",
) -> _Array_fc_1d: ...
def savgol_filter(
    x: Untyped,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: str = "interp",
    cval: float = 0.0,
) -> UntypedArray: ...
