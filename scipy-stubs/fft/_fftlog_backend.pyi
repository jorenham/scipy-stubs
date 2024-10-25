import numpy as np
import numpy.typing as npt
from optype import CanBool
from scipy._typing import AnyInt, AnyReal
from ._fftlog import fht, fhtoffset, ifht

__all__ = ["fht", "fhtoffset", "ifht"]

def fhtcoeff(
    n: AnyInt,
    dln: AnyReal,
    mu: AnyReal,
    offset: AnyReal = 0.0,
    bias: AnyReal = 0.0,
    inverse: CanBool = False,
) -> npt.NDArray[np.complex128]: ...
