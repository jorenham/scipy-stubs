from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

__all__ = ["expm_cond", "expm_frechet"]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

@overload
def expm_frechet(
    A: npt.ArrayLike,
    E: npt.ArrayLike,
    method: Literal["SPS", "blockEnlarge"] | None = None,
    compute_expm: Literal[True] = True,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def expm_frechet(
    A: npt.ArrayLike,
    E: npt.ArrayLike,
    method: Literal["SPS", "blockEnlarge"] | None = None,
    *,
    compute_expm: Literal[False],
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def expm_frechet(
    A: npt.ArrayLike,
    E: npt.ArrayLike,
    method: Literal["SPS", "blockEnlarge"] | None,
    compute_expm: Literal[False],
    /,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
def expm_cond(A: npt.ArrayLike, check_finite: bool = True) -> np.float64: ...
