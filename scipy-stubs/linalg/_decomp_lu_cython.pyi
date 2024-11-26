from typing import TypeVar

import numpy as np
import optype.numpy as onp

# this name was chosen to match `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", bound=np.float32 | np.float64 | np.complex64 | np.complex128)

def lu_dispatcher(
    a: onp.ArrayND[_LapackT],
    u: onp.ArrayND[_LapackT],
    piv: onp.ArrayND[np.int32 | np.int64],
    permute_l: bool,
) -> None: ...
