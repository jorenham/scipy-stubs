from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, type_check_only

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import Alternative, AnyReal, Seed
from ._common import ConfidenceInterval

__all__ = ["bootstrap", "monte_carlo_test", "permutation_test"]

_BootstrapMethod: TypeAlias = Literal["percentile", "basic", "BCa"]

@type_check_only
class _RVSCallable(Protocol):
    def __call__(self, /, *, size: tuple[int, ...]) -> npt.NDArray[np.floating[Any]]: ...

@dataclass
class BootstrapResult:
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: npt.NDArray[np.float64]
    standard_error: float | npt.NDArray[np.float64]

@dataclass
class MonteCarloTestResult:
    statistic: float | npt.NDArray[np.float64]
    pvalue: float | npt.NDArray[np.float64]
    null_distribution: npt.NDArray[np.float64]

@dataclass
class PowerResult:
    power: float | npt.NDArray[np.float64]
    pvalues: float | npt.NDArray[np.float64]

@dataclass
class PermutationTestResult:
    statistic: float | npt.NDArray[np.float64]
    pvalue: float | npt.NDArray[np.float64]
    null_distribution: npt.NDArray[np.float64]

@dataclass
class ResamplingMethod:
    n_resamples: int = 999
    batch: int | None = None

@dataclass
class MonteCarloMethod(ResamplingMethod):
    rvs: object | None = None

@dataclass
class PermutationMethod(ResamplingMethod):
    random_state: object | None = None

@dataclass
class BootstrapMethod(ResamplingMethod):
    random_state: object | None = None
    method: str = "BCa"

def bootstrap(
    data: _ArrayLikeFloat_co,
    statistic: Callable[[Any], AnyReal],
    *,
    n_resamples: int = 9999,
    batch: int | None = None,
    vectorized: bool | None = None,
    paired: bool = False,
    axis: int = 0,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    method: _BootstrapMethod = "BCa",
    bootstrap_result: BootstrapResult | None = None,
    random_state: Seed | None = None,
) -> BootstrapResult: ...
def monte_carlo_test(
    data: _ArrayLikeFloat_co,
    rvs: _RVSCallable,
    statistic: Callable[[Any], AnyReal],
    *,
    vectorized: bool | None = None,
    n_resamples: int = 9999,
    batch: int | None = None,
    alternative: Alternative = "two-sided",
    axis: int = 0,
) -> MonteCarloTestResult: ...
def power(
    test: Callable[..., float | np.floating[Any]],
    rvs: _RVSCallable,
    n_observations: Sequence[int] | Sequence[npt.NDArray[np.integer[Any]]],
    *,
    significance: _ArrayLikeFloat_co = 0.01,
    kwargs: Mapping[str, object] | None = None,
    vectorized: bool | None = None,
    n_resamples: int = 10000,
    batch: int | None = None,
) -> PowerResult: ...
def permutation_test(
    data: _ArrayLikeFloat_co,
    statistic: Callable[..., AnyReal],
    *,
    permutation_type: str = "independent",
    vectorized: bool | None = None,
    n_resamples: int = 9999,
    batch: int | None = None,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    random_state: Seed | None = None,
) -> PermutationTestResult: ...
