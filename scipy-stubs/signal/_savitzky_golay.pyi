from ._arraytools import axis_slice as axis_slice
from scipy._lib._util import float_factorial as float_factorial
from scipy._typing import Untyped
from scipy.linalg import lstsq as lstsq
from scipy.ndimage import convolve1d as convolve1d

def savgol_coeffs(
    window_length, polyorder, deriv: int = 0, delta: float = 1.0, pos: Untyped | None = None, use: str = "conv"
) -> Untyped: ...
def savgol_filter(
    x, window_length, polyorder, deriv: int = 0, delta: float = 1.0, axis: int = -1, mode: str = "interp", cval: float = 0.0
) -> Untyped: ...
