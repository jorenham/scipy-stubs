from typing import Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["expm_cond", "expm_frechet"]

_Method: TypeAlias = Literal["SPS", "blockEnlarge"]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]

###

@overload
def expm_frechet(
    A: onp.ToComplex2D,
    E: onp.ToComplex2D,
    method: _Method | None = None,
    compute_expm: Literal[True] = True,
    check_finite: bool = True,
) -> tuple[_Float2D, _Float2D] | tuple[_Float2D | _Complex2D, _Complex2D]: ...
@overload
def expm_frechet(
    A: onp.ToComplex2D,
    E: onp.ToComplex2D,
    method: _Method | None,
    compute_expm: Literal[False],
    check_finite: bool = True,
) -> tuple[_Float2D, _Float2D] | tuple[_Float2D | _Complex2D, _Complex2D]: ...
@overload
def expm_frechet(
    A: onp.ToComplex2D,
    E: onp.ToComplex2D,
    method: _Method | None = None,
    *,
    compute_expm: Literal[False],
    check_finite: bool = True,
) -> tuple[_Float2D, _Float2D] | tuple[_Float2D | _Complex2D, _Complex2D]: ...

#
def expm_cond(A: onp.ToComplex2D, check_finite: bool = True) -> np.float64: ...
