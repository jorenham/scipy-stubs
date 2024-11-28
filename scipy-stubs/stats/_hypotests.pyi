from dataclasses import dataclass
from collections.abc import Callable
from typing import Concatenate, Generic, Literal, NamedTuple, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Alternative, NanPolicy
from ._common import ConfidenceInterval
from ._stats_py import SignificanceResult

__all__ = [
    "barnard_exact",
    "boschloo_exact",
    "cramervonmises",
    "cramervonmises_2samp",
    "epps_singleton_2samp",
    "poisson_means_test",
    "somersd",
    "tukey_hsd",
]

_FloatOrArray: TypeAlias = float | np.float64 | onp.ArrayND[np.float64]
_FloatOrArrayT = TypeVar("_FloatOrArrayT", bound=_FloatOrArray, default=_FloatOrArray)

class Epps_Singleton_2sampResult(NamedTuple, Generic[_FloatOrArrayT]):
    statistic: _FloatOrArrayT
    pvalue: _FloatOrArrayT

class CramerVonMisesResult(Generic[_FloatOrArrayT]):
    statistic: _FloatOrArrayT
    pvalue: _FloatOrArrayT
    def __init__(self, /, statistic: _FloatOrArrayT, pvalue: _FloatOrArrayT) -> None: ...

class TukeyHSDResult:
    statistic: onp.ArrayND[np.float64]
    pvalue: onp.ArrayND[np.float64]
    def __init__(
        self,
        /,
        statistic: onp.ArrayND[np.float64],
        pvalue: onp.ArrayND[np.float64],
        _nobs: int,
        _ntreatments: int,
        _stand_err: float,
    ) -> None: ...
    def confidence_interval(self, /, confidence_level: float = 0.95) -> ConfidenceInterval: ...

@dataclass
class SomersDResult:
    statistic: float | np.float64
    pvalue: float | np.float64
    table: onp.Array2D[np.float64]

@dataclass
class BarnardExactResult:
    statistic: float | np.float64
    pvalue: float | np.float64

@dataclass
class BoschlooExactResult:
    statistic: float | np.float64
    pvalue: float | np.float64

def epps_singleton_2samp(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    t: onp.ToFloat1D = (0.4, 0.8),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Epps_Singleton_2sampResult: ...
def poisson_means_test(
    k1: int,
    n1: float,
    k2: int,
    n2: float,
    *,
    diff: float = 0,
    alternative: Alternative = "two-sided",
) -> SignificanceResult[np.float64]: ...
def cramervonmises(
    rvs: onp.ToFloat1D,
    cdf: str | Callable[Concatenate[float, ...], float | np.float64 | np.float32],
    args: tuple[onp.ToFloat, ...] = (),
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...
def cramervonmises_2samp(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> CramerVonMisesResult: ...
def somersd(
    x: onp.ToFloat1D | onp.ToFloat2D,
    y: onp.ToFloat1D | None = None,
    alternative: Alternative = "two-sided",
) -> SomersDResult: ...
def barnard_exact(
    table: onp.ToInt2D,
    alternative: Alternative = "two-sided",
    pooled: bool = True,
    n: int = 32,
) -> BarnardExactResult: ...
def boschloo_exact(table: onp.ToInt2D, alternative: Alternative = "two-sided", n: int = 32) -> BoschlooExactResult: ...
def tukey_hsd(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND) -> TukeyHSDResult: ...
