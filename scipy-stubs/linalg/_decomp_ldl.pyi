from typing import Any, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["ldl"]

_IntP1D: TypeAlias = onp.Array1D[np.intp]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.inexact[Any]]

###

@overload
def ldl(
    A: onp.ToFloat2D,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float2D, _IntP1D]: ...
@overload
def ldl(
    A: onp.ToComplex2D,
    lower: onp.ToBool = True,
    hermitian: onp.ToBool = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Complex2D, _IntP1D]: ...
