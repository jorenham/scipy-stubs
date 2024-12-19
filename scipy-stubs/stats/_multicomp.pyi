from dataclasses import dataclass
from typing import Any

import numpy as np
import optype.numpy as onp
from scipy._typing import Alternative, ToRNG
from ._common import ConfidenceInterval

__all__ = ["dunnett"]

@dataclass
class DunnettResult:
    statistic: onp.Array1D[np.float64]
    pvalue: onp.Array1D[np.float64]

    _alternative: Alternative
    _rho: onp.Array2D[np.float64]
    _df: int
    _std: np.float64
    _mean_samples: onp.Array1D[np.float64]
    _mean_control: np.float64  # incorrectly annotated as `ndarray` at runtime
    _n_samples: onp.Array1D[np.int_]
    _n_control: int
    _rng: np.random.Generator | np.random.RandomState

    _ci: ConfidenceInterval | None = None
    _ci_cl: float | np.floating[Any] | None = None

    def confidence_interval(self, /, confidence_level: float | np.floating[Any] = 0.95) -> ConfidenceInterval: ...

def dunnett(
    *samples: onp.ToFloat1D,
    control: onp.ToFloat1D,
    alternative: Alternative = "two-sided",
    rng: ToRNG = None,
) -> DunnettResult: ...
