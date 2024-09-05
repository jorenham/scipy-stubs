from typing import TypeVar

import numpy as np
import numpy.typing as npt

# this name was chosen to match `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", bound=np.float32 | np.float64 | np.complex64 | np.complex128)

def lu_dispatcher(
    a: npt.NDArray[_LapackT],
    u: npt.NDArray[_LapackT],
    piv: npt.NDArray[np.int32 | np.int64],
    permute_l: bool,
) -> None: ...
