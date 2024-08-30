from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy._typing import Untyped, UntypedArray
from . import distributions as distributions
from ._common import ConfidenceInterval as ConfidenceInterval
from ._continuous_distns import norm as norm

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

class Epps_Singleton_2sampResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

@dataclass
class SomersDResult:
    statistic: float
    pvalue: float
    table: UntypedArray

@dataclass
class BarnardExactResult:
    statistic: float
    pvalue: float

@dataclass
class BoschlooExactResult:
    statistic: float
    pvalue: float

class TukeyHSDResult:
    statistic: Untyped
    pvalue: Untyped
    def __init__(self, statistic, pvalue, _nobs, _ntreatments, _stand_err) -> None: ...
    def confidence_interval(self, confidence_level: float = 0.95) -> Untyped: ...

class CramerVonMisesResult:
    statistic: Untyped
    pvalue: Untyped
    def __init__(self, statistic, pvalue) -> None: ...

def epps_singleton_2samp(x, y, t=(0.4, 0.8)) -> Untyped: ...
def poisson_means_test(k1, n1, k2, n2, *, diff: int = 0, alternative: str = "two-sided") -> Untyped: ...
def cramervonmises(rvs, cdf, args=()) -> Untyped: ...
def somersd(x, y: Untyped | None = None, alternative: str = "two-sided") -> Untyped: ...
def barnard_exact(table, alternative: str = "two-sided", pooled: bool = True, n: int = 32) -> Untyped: ...
def boschloo_exact(table, alternative: str = "two-sided", n: int = 32) -> Untyped: ...
def cramervonmises_2samp(x, y, method: str = "auto") -> Untyped: ...
def tukey_hsd(*args) -> Untyped: ...
