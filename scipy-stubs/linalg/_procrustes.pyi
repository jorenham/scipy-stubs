from typing import TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = ["orthogonal_procrustes"]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

###

@overload
def orthogonal_procrustes(
    A: onp.ToFloat2D,
    B: onp.ToFloat2D,
    check_finite: op.CanBool = True,
) -> tuple[onp.Array2D[_Float], _Float]: ...
@overload
def orthogonal_procrustes(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    check_finite: onp.ToBool = True,
) -> tuple[onp.Array2D[_Float | _Complex], _Float]: ...
