from typing import Any

import numpy as np
from scipy.stats import qmc

_f8_xd: np.ndarray[Any, np.dtype[np.float64]]
qmc.scale(_f8_xd, 0, 1)

_f8_nd: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
qmc.scale(_f8_nd, 0, 1)

_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]
qmc.scale(_f8_2d, 0, 1)
