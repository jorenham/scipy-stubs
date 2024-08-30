from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy._typing as spt

__all__ = ["find_best_blas_type", "get_blas_funcs"]

def find_best_blas_type(
    arrays: Sequence[npt.NDArray[np.generic]] = (),
    dtype: npt.DTypeLike | None = None,
) -> (
    # see `scipy.linalg.blas._type_conv`
    tuple[Literal["s"], np.dtypes.Float32DType, bool]
    | tuple[Literal["f"], np.dtypes.Float64DType, bool]
    | tuple[Literal["c"], np.dtypes.Complex64DType, bool]
    | tuple[Literal["z"], np.dtypes.Complex128DType, bool]
): ...
def get_blas_funcs(
    names: Iterable[str] | str,
    arrays: Sequence[npt.NDArray[np.generic]] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: Literal[True, False, "preferred"] = False,
) -> list[spt._FortranFunction] | spt._FortranFunction: ...
