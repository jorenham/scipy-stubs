from typing import TypeVar

import numpy as np
import numpy.typing as npt

__all__ = ["lu_decompose", "lu_dispatcher"]

# this name was chosen to match `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", bound=np.float32 | np.float64 | np.complex64 | np.complex128)

def lu_decompose(a: npt.NDArray[_LapackT], lu: npt.NDArray[_LapackT], perm: npt.NDArray[np.int_], permute_l: bool) -> None: ...
def lu_dispatcher(a: npt.NDArray[_LapackT], lu: npt.NDArray[_LapackT], perm: npt.NDArray[np.int_], permute_l: bool) -> None: ...
