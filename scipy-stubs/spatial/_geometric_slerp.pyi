import numpy.typing as npt
import scipy._typing as spt

__all__ = ["geometric_slerp"]

def geometric_slerp(start: npt.ArrayLike, end: npt.ArrayLike, t: npt.ArrayLike, tol: float = 1e-07) -> spt.UntypedArray: ...
