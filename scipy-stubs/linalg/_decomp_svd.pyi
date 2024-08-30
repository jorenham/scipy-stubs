from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import scipy._typing as spt

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_Array_f_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[npt.NBitBase]]]
_Array_f_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating[npt.NBitBase]]]

@overload
def svd(
    a: npt.ArrayLike,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gesdd", "gesvd"] = "gesdd",
) -> tuple[_Array_f_2d, _Array_f_1d, _Array_f_2d]: ...
@overload
def svd(
    a: npt.ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gesdd", "gesvd"] = "gesdd",
) -> _Array_f_1d: ...
@overload
def svd(
    a: npt.ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[False],
    /,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gesdd", "gesvd"] = "gesdd",
) -> _Array_f_1d: ...
def svdvals(a: npt.ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> _Array_f_1d: ...
def diagsvd(s: npt.ArrayLike, M: spt.AnyInt, N: spt.AnyInt) -> _Array_f_2d: ...
def orth(A: npt.ArrayLike, rcond: spt.AnyReal | None = None) -> _Array_f_2d: ...
def null_space(
    A: npt.ArrayLike,
    rcond: spt.AnyReal | None = None,
    *,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: Literal["gesdd", "gesvd"] = "gesdd",
) -> _Array_f_2d: ...
def subspace_angles(A: npt.ArrayLike, B: npt.ArrayLike) -> _Array_f_1d: ...
