from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import optype.numpy as onp
from scipy._typing import Alternative, Seed
from ._common import ConfidenceInterval

__all__ = ["dunnett"]

@dataclass
class DunnettResult:
    statistic: onp.ArrayND[np.float64]
    pvalue: onp.ArrayND[np.float64]
    _alternative: Literal["two-sided", "less", "greater"]
    _rho: onp.ArrayND[np.float64]
    _df: int
    _std: float
    _mean_samples: onp.ArrayND[np.float64]
    _mean_control: onp.ArrayND[np.float64]
    _n_samples: onp.ArrayND[np.int_]
    _n_control: int
    _rng: Seed
    _ci: ConfidenceInterval | None = None
    _ci_cl: float | np.floating[Any] | np.integer[Any] | None = None

    def confidence_interval(self, /, confidence_level: onp.ToFloat = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: onp.ToFloat1D,
    control: onp.ToFloat1D,
    alternative: Alternative = "two-sided",
    random_state: Seed | None = None,
) -> DunnettResult: ...
