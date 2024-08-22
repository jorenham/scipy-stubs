from dataclasses import dataclass

import numpy as np

from ._common import ConfidenceInterval as ConfidenceInterval
from ._warnings_errors import DegenerateDataWarning as DegenerateDataWarning
from scipy._lib._array_api import (
    array_namespace as array_namespace,
    is_numpy as is_numpy,
    xp_moveaxis_to_end as xp_moveaxis_to_end,
)
from scipy._lib._util import check_random_state as check_random_state, rng_integers as rng_integers
from scipy._typing import Untyped
from scipy.special import comb as comb, factorial as factorial, ndtr as ndtr, ndtri as ndtri

@dataclass
class BootstrapResult:
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray

def bootstrap(
    data,
    statistic,
    *,
    n_resamples: int = 9999,
    batch: Untyped | None = None,
    vectorized: Untyped | None = None,
    paired: bool = False,
    axis: int = 0,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
    method: str = "BCa",
    bootstrap_result: Untyped | None = None,
    random_state: Untyped | None = None,
) -> Untyped: ...
@dataclass
class MonteCarloTestResult:
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray

def monte_carlo_test(
    data,
    rvs,
    statistic,
    *,
    vectorized: Untyped | None = None,
    n_resamples: int = 9999,
    batch: Untyped | None = None,
    alternative: str = "two-sided",
    axis: int = 0,
) -> Untyped: ...
@dataclass
class PowerResult:
    power: float | np.ndarray
    pvalues: float | np.ndarray

def power(
    test,
    rvs,
    n_observations,
    *,
    significance: float = 0.01,
    vectorized: Untyped | None = None,
    n_resamples: int = 10000,
    batch: Untyped | None = None,
    kwargs: Untyped | None = None,
) -> Untyped: ...
@dataclass
class PermutationTestResult:
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray

def permutation_test(
    data,
    statistic,
    *,
    permutation_type: str = "independent",
    vectorized: Untyped | None = None,
    n_resamples: int = 9999,
    batch: Untyped | None = None,
    alternative: str = "two-sided",
    axis: int = 0,
    random_state: Untyped | None = None,
) -> Untyped: ...
@dataclass
class ResamplingMethod:
    n_resamples: int = ...
    batch: int = ...

@dataclass
class MonteCarloMethod(ResamplingMethod):
    rvs: object = ...

@dataclass
class PermutationMethod(ResamplingMethod):
    random_state: object = ...

@dataclass
class BootstrapMethod(ResamplingMethod):
    random_state: object = ...
    method: str = ...
