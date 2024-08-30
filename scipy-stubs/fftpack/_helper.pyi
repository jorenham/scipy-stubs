from numpy.fft import fftfreq, fftshift, ifftshift  # noqa: ICN003
from scipy._typing import Untyped

__all__ = ["fftfreq", "fftshift", "ifftshift", "next_fast_len", "rfftfreq"]

def rfftfreq(n, d: float = 1.0) -> Untyped: ...
def next_fast_len(target) -> Untyped: ...
