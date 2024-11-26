from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["sqrtm"]

_Array_fc_2d: TypeAlias = onp.Array2D[np.inexact[Any]]

class SqrtmError(np.linalg.LinAlgError): ...

@overload
def sqrtm(A: npt.ArrayLike, disp: Literal[True] = True, blocksize: int = 64) -> _Array_fc_2d: ...
@overload
def sqrtm(A: npt.ArrayLike, disp: Literal[False], blocksize: int = 64) -> tuple[_Array_fc_2d, float | np.float64]: ...
