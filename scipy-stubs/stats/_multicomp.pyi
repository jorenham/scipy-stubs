from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from scipy import stats as stats
from scipy._lib._util import DecimalNumber as DecimalNumber, SeedType as SeedType
from scipy.optimize import minimize_scalar as minimize_scalar
from scipy.stats._common import ConfidenceInterval as ConfidenceInterval
from scipy.stats._qmc import check_random_state as check_random_state

@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray
    def confidence_interval(self, confidence_level: DecimalNumber = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    random_state: SeedType = None,
) -> DunnettResult: ...
