from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyReal, Seed
from ._common import ConfidenceInterval

__all__ = ["dunnett"]

@dataclass
class DunnettResult:
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]
    def confidence_interval(self, confidence_level: AnyReal = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    random_state: Seed | None = None,
) -> DunnettResult: ...
