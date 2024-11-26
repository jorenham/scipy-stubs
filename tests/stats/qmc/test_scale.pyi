from typing import Any

import numpy as np
import optype.numpy as onp
from scipy.stats import qmc

_f8_xd: np.ndarray[Any, np.dtype[np.float64]]
qmc.scale(_f8_xd, 0, 1)

_f8_nd: onp.ArrayND[np.float64]
qmc.scale(_f8_nd, 0, 1)

_f8_2d: onp.Array2D[np.float64]
qmc.scale(_f8_2d, 0, 1)
