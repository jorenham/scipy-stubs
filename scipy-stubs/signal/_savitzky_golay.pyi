from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]
_Mode: TypeAlias = Literal["mirror", "constant", "nearest", "wrap", "interp"]

def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: Literal["conv", "dot"] = "conv",
) -> _Array_fc_1d: ...
def savgol_filter(
    x: npt.ArrayLike,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: _Mode = "interp",
    cval: float = 0.0,
) -> npt.NDArray[np.float32 | np.float64]: ...
