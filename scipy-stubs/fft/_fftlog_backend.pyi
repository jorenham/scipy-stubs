import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from ._fftlog import fht, fhtoffset, ifht

__all__ = ["fht", "fhtoffset", "ifht"]

def fhtcoeff(
    n: onp.ToInt,
    dln: onp.ToFloat,
    mu: onp.ToFloat,
    offset: onp.ToFloat = 0.0,
    bias: onp.ToFloat = 0.0,
    inverse: op.CanBool = False,
) -> npt.NDArray[np.complex128]: ...
