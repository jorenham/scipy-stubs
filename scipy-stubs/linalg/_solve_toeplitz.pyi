from typing import TypeAlias

import numpy as np
import numpy.typing as npt

_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64 | np.complex128]]

def levinson(a: npt.NDArray[np.float64 | np.complex128], b: npt.NDArray[np.float64 | np.complex128]) -> tuple[_Array_fc_1d]: ...
