from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats as stats
from scipy._lib._util import DecimalNumber as DecimalNumber, SeedType as SeedType
from scipy.stats._common import ConfidenceInterval as ConfidenceInterval

__all__ = ["dunnett"]

@dataclass
class DunnettResult:
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]
    def confidence_interval(self, confidence_level: DecimalNumber = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    random_state: SeedType = None,
) -> DunnettResult: ...
