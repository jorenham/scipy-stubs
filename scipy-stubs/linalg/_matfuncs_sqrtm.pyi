from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

__all__ = ["sqrtm"]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

class SqrtmError(np.linalg.LinAlgError): ...

@overload
def sqrtm(A: npt.ArrayLike, disp: Literal[True] = True, blocksize: int = 64) -> _Array_fc_2d: ...
@overload
def sqrtm(A: npt.ArrayLike, disp: Literal[False], blocksize: int = 64) -> tuple[_Array_fc_2d, float | np.float64]: ...
