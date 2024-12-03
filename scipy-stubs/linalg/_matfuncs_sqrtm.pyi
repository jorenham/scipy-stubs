from typing import Any, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["sqrtm"]

_Inexact2D: TypeAlias = onp.Array2D[np.floating[Any]] | onp.Array2D[np.complexfloating[Any, Any]]

_Falsy: TypeAlias = Literal[False, 0]
_Truthy: TypeAlias = Literal[True, 1]

###

class SqrtmError(np.linalg.LinAlgError): ...  # undocumented

# NOTE: The output dtype (floating or complex) depends on the sign of the values, so this is the best we can do.
@overload
def sqrtm(A: onp.ToComplex2D, disp: _Truthy = True, blocksize: onp.ToJustInt = 64) -> _Inexact2D: ...
@overload
def sqrtm(A: onp.ToComplex2D, disp: _Falsy, blocksize: onp.ToJustInt = 64) -> tuple[_Inexact2D, np.floating[Any]]: ...
