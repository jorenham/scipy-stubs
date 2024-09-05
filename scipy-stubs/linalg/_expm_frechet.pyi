from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onpt

__all__ = ["expm_cond", "expm_frechet"]

_Method: TypeAlias = Literal["SPS", "blockEnlarge"]

_ArrayLike_2d_fc: TypeAlias = onpt.AnyNumberArray | Sequence[Sequence[complex | np.number[Any]]]
_Array_2d_f8: TypeAlias = onpt.Array[tuple[int, int], np.float64]
_Array_2d_c16: TypeAlias = onpt.Array[tuple[int, int], np.complex128]

@overload
def expm_frechet(
    A: _ArrayLike_2d_fc,
    E: _ArrayLike_2d_fc,
    method: _Method | None = None,
    compute_expm: Literal[True] = True,
    check_finite: bool = True,
) -> tuple[_Array_2d_f8, _Array_2d_f8] | tuple[_Array_2d_f8 | _Array_2d_c16, _Array_2d_c16]: ...
@overload
def expm_frechet(
    A: _ArrayLike_2d_fc,
    E: _ArrayLike_2d_fc,
    method: _Method | None,
    compute_expm: Literal[False],
    check_finite: bool = True,
) -> tuple[_Array_2d_f8, _Array_2d_f8] | tuple[_Array_2d_f8 | _Array_2d_c16, _Array_2d_c16]: ...
@overload
def expm_frechet(
    A: _ArrayLike_2d_fc,
    E: _ArrayLike_2d_fc,
    method: _Method | None = None,
    *,
    compute_expm: Literal[False],
    check_finite: bool = True,
) -> tuple[_Array_2d_f8, _Array_2d_f8] | tuple[_Array_2d_f8 | _Array_2d_c16, _Array_2d_c16]: ...
def expm_cond(A: _ArrayLike_2d_fc, check_finite: bool = True) -> np.float64: ...
