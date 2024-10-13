from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy._typing import Alternative, Untyped
from ._censored_data import CensoredData
from ._common import ConfidenceInterval

__all__ = ["ecdf", "logrank"]

@dataclass
class EmpiricalDistributionFunction:
    quantiles: npt.NDArray[np.float64]
    probabilities: npt.NDArray[np.float64]
    _n: npt.NDArray[np.int_]
    _d: npt.NDArray[np.int_]
    _sf: npt.NDArray[np.float64]
    _kind: Literal["cdf", "sf"]

    def __init__(
        self,
        /,
        q: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
        n: npt.NDArray[np.int_],
        d: npt.NDArray[np.int_],
        kind: Literal["cdf", "sf"],
    ) -> None: ...
    def evaluate(self, /, x: npt.NDArray[np.float64]) -> Untyped: ...
    def plot(self, /, ax: object | None = None, **matplotlib_kwargs: object) -> list[Any]: ...
    def confidence_interval(
        self,
        /,
        confidence_level: float = 0.95,
        *,
        method: Literal["linear", "log-log"] = "linear",
    ) -> ConfidenceInterval: ...

@dataclass
class ECDFResult:
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction

    def __init__(
        self,
        /,
        q: npt.NDArray[np.float64],
        cdf: npt.NDArray[np.float64],
        sf: npt.NDArray[np.float64],
        n: npt.NDArray[np.int_],
        d: npt.NDArray[np.int_],
    ) -> None: ...

@dataclass
class LogRankResult:
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]

def ecdf(sample: npt.ArrayLike | CensoredData) -> ECDFResult: ...
def logrank(
    x: npt.ArrayLike | CensoredData,
    y: npt.ArrayLike | CensoredData,
    alternative: Alternative = "two-sided",
) -> LogRankResult: ...
