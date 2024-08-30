from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype as op

__all__ = ["cossin"]

_Array_f_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[npt.NBitBase]]]
_Array_f_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating[npt.NBitBase]]]
_Array_c_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complexfloating[npt.NBitBase, npt.NBitBase]]]

@overload
def cossin(
    X: npt.ArrayLike | op.CanIter[op.CanNext[npt.ArrayLike]],
    p: op.typing.AnyInt | None = None,
    q: op.typing.AnyInt | None = None,
    separate: Literal[False] = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> tuple[_Array_f_2d, _Array_f_2d, _Array_f_2d] | tuple[_Array_c_2d, _Array_f_2d, _Array_c_2d]: ...
@overload
def cossin(
    X: npt.ArrayLike | op.CanIter[op.CanNext[npt.ArrayLike]],
    p: op.typing.AnyInt | None = None,
    q: op.typing.AnyInt | None = None,
    *,
    separate: Literal[True],
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> (
    tuple[tuple[_Array_f_2d, _Array_f_2d], _Array_f_1d, tuple[_Array_f_2d, _Array_f_2d]]
    | tuple[tuple[_Array_c_2d, _Array_c_2d], _Array_f_1d, tuple[_Array_c_2d, _Array_c_2d]]
): ...
