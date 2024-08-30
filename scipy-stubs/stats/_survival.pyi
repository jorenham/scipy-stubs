from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import interpolate as interpolate, special as special, stats as stats
from scipy._typing import Untyped
from scipy.stats._censored_data import CensoredData as CensoredData
from scipy.stats._common import ConfidenceInterval as ConfidenceInterval

@dataclass
class EmpiricalDistributionFunction:
    quantiles: npt.NDArray[np.float64]
    probabilities: npt.NDArray[np.float64]
    def __init__(self, q, p, n, d, kind) -> None: ...
    def evaluate(self, x) -> Untyped: ...
    def plot(self, ax: Untyped | None = None, **matplotlib_kwargs: Untyped) -> Untyped: ...
    def confidence_interval(self, confidence_level: float = 0.95, *, method: str = "linear") -> Untyped: ...

@dataclass
class ECDFResult:
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction
    def __init__(self, q, cdf, sf, n, d) -> None: ...

def ecdf(sample: npt.ArrayLike | CensoredData) -> ECDFResult: ...
@dataclass
class LogRankResult:
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]

def logrank(
    x: npt.ArrayLike | CensoredData,
    y: npt.ArrayLike | CensoredData,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> LogRankResult: ...
