from dataclasses import dataclass

from scipy._typing import Untyped, UntypedArray
from ._common import ConfidenceInterval

__all__ = ["bootstrap", "monte_carlo_test", "permutation_test"]

@dataclass
class BootstrapResult:
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: UntypedArray
    standard_error: float | UntypedArray

@dataclass
class MonteCarloTestResult:
    statistic: float | UntypedArray
    pvalue: float | UntypedArray
    null_distribution: UntypedArray

@dataclass
class PowerResult:
    power: float | UntypedArray
    pvalues: float | UntypedArray

@dataclass
class PermutationTestResult:
    statistic: float | UntypedArray
    pvalue: float | UntypedArray
    null_distribution: UntypedArray

@dataclass
class ResamplingMethod:
    n_resamples: int = ...
    batch: int = ...

@dataclass
class MonteCarloMethod(ResamplingMethod):
    rvs: Untyped = ...

@dataclass
class PermutationMethod(ResamplingMethod):
    random_state: Untyped = ...

@dataclass
class BootstrapMethod(ResamplingMethod):
    random_state: Untyped = ...
    method: str = ...

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
