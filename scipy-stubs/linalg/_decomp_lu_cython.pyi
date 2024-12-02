from typing import TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp

_ST = TypeVar("_ST", bound=np.float32 | np.float64 | np.complex64 | np.complex128)
_Int1D: TypeAlias = onp.Array1D[np.int32 | np.int64]

###

def lu_dispatcher(a: onp.Array2D[_ST], u: onp.Array2D[_ST], piv: _Int1D, permute_l: onp.ToBool) -> None: ...
