from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import scipy._typing as spt

__all__ = ["find_best_blas_type", "get_blas_funcs"]

# see `scipy.linalg.blas._type_conv`
def find_best_blas_type(
    arrays: Sequence[onp.ArrayND] = (),
    dtype: npt.DTypeLike | None = None,
) -> (
    tuple[Literal["s"], np.dtype[np.float32], bool]
    | tuple[Literal["f"], np.dtype[np.float64], bool]
    | tuple[Literal["c"], np.dtype[np.complex64], bool]
    | tuple[Literal["z"], np.dtype[np.complex128], bool]
): ...
def get_blas_funcs(
    names: Iterable[str] | str,
    arrays: Sequence[onp.ArrayND] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: Literal["preferred"] | bool = False,
) -> list[spt._FortranFunction] | spt._FortranFunction: ...
