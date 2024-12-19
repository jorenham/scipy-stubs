from typing import NamedTuple

import numpy as np

class ConfidenceInterval(NamedTuple):
    low: float | np.float64
    high: float | np.float64
