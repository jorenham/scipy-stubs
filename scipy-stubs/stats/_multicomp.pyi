from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy._typing import Alternative, AnyReal, Seed
from ._common import ConfidenceInterval

__all__ = ["dunnett"]

@dataclass
class DunnettResult:
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]
    _alternative: Literal["two-sided", "less", "greater"]
    _rho: npt.NDArray[np.float64]
    _df: int
    _std: float
    _mean_samples: npt.NDArray[np.float64]
    _mean_control: npt.NDArray[np.float64]
    _n_samples: npt.NDArray[np.int_]
    _n_control: int
    _rng: Seed
    _ci: ConfidenceInterval | None = None
    _ci_cl: float | np.floating[Any] | np.integer[Any] | None = None

    def confidence_interval(self, /, confidence_level: AnyReal = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Alternative = "two-sided",
    random_state: Seed | None = None,
) -> DunnettResult: ...
