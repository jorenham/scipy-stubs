from scipy._typing import Untyped
from ._fftlog_backend import fhtoffset

__all__ = ["fht", "fhtoffset", "ifht"]

def fht(a: Untyped, dln: Untyped, mu: Untyped, offset: float = 0.0, bias: float = 0.0) -> Untyped: ...
def ifht(A: Untyped, dln: Untyped, mu: Untyped, offset: float = 0.0, bias: float = 0.0) -> Untyped: ...
