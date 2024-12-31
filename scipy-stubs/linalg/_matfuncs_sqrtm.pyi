from typing import Any, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

__all__ = ["sqrtm"]

_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

###

class SqrtmError(np.linalg.LinAlgError): ...  # undocumented

# NOTE: The output dtype (floating or complex) depends on the sign of the values, so this is the best we can do.
@overload
def sqrtm(A: onp.ToComplex2D, disp: Truthy = True, blocksize: onp.ToJustInt = 64) -> _Inexact2D: ...
@overload
def sqrtm(A: onp.ToComplex2D, disp: Falsy, blocksize: onp.ToJustInt = 64) -> tuple[_Inexact2D, np.floating[Any]]: ...
