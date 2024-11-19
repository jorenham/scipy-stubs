import numpy as np
import optype.numpy as onp
from numpy.fft import fftfreq, fftshift, ifftshift
from scipy._typing import AnyInt, AnyReal

__all__ = ["fftfreq", "fftshift", "ifftshift", "next_fast_len", "rfftfreq"]

def rfftfreq(n: AnyInt, d: AnyReal = 1.0) -> onp.Array[tuple[int], np.float64]: ...
def next_fast_len(target: AnyInt) -> int: ...
