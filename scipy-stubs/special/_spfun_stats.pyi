from typing import overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["multigammaln"]

@overload
def multigammaln(a: onp.ToFloat, d: onp.ToInt) -> np.float64: ...
@overload
def multigammaln(a: onp.ToFloatND, d: onp.ToInt) -> npt.NDArray[np.float64]: ...
