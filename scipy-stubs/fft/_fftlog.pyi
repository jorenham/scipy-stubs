from scipy._typing import Untyped

from scipy._lib.uarray import Dispatchable as Dispatchable
from ._fftlog_backend import fhtoffset as fhtoffset

def fht(a, dln, mu, offset: float = 0.0, bias: float = 0.0) -> Untyped: ...
def ifht(A, dln, mu, offset: float = 0.0, bias: float = 0.0) -> Untyped: ...
