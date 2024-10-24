from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import Alternative, AnyReal
from ._censored_data import CensoredData
from ._common import ConfidenceInterval

__all__ = ["ecdf", "logrank"]

_EDFKind: TypeAlias = Literal["cdf", "sf"]
_CIMethod: TypeAlias = Literal["linear", "log-log"]

_VectorInt: TypeAlias = onpt.Array[tuple[int], np.int_]
_VectorFloat: TypeAlias = onpt.Array[tuple[int], np.float64]

_KwargsT = TypeVar("_KwargsT")
_KwargsT_contra = TypeVar("_KwargsT_contra", contravariant=True)
_LineT = TypeVar("_LineT")

_SampleData: TypeAlias = _ArrayLikeFloat_co | CensoredData

@type_check_only
class _CanStep(Protocol[_KwargsT_contra, _LineT]):
    def step(self, x: _VectorFloat, y: _VectorFloat, /, **kwargs: _KwargsT_contra) -> list[_LineT]: ...

###

@dataclass
class EmpiricalDistributionFunction:
    # NOTE: the order of attributes matters
    quantiles: _VectorFloat
    probabilities: _VectorFloat
    _n: _VectorInt
    _d: _VectorInt
    _sf: _VectorFloat
    _kind: _EDFKind

    def __init__(self, /, q: _VectorFloat, p: _VectorFloat, n: _VectorInt, d: _VectorInt, kind: _EDFKind) -> None: ...
    def evaluate(self, /, x: _ArrayLikeFloat_co) -> npt.NDArray[np.float64]: ...
    @overload
    def plot(self, /, ax: None = None, **kwds: object) -> list[Any]: ...
    @overload
    def plot(self, /, ax: _CanStep[_KwargsT, _LineT], **kwds: _KwargsT) -> list[_LineT]: ...
    def confidence_interval(self, /, confidence_level: AnyReal = 0.95, *, method: _CIMethod = "linear") -> ConfidenceInterval: ...

@dataclass
class ECDFResult:
    cdf: Final[EmpiricalDistributionFunction]
    sf: Final[EmpiricalDistributionFunction]

    def __init__(self, /, q: _VectorFloat, cdf: _VectorFloat, sf: _VectorFloat, n: _VectorInt, d: _VectorInt) -> None: ...

@dataclass
class LogRankResult:
    statistic: np.float64
    pvalue: np.float64

def ecdf(sample: _SampleData) -> ECDFResult: ...
def logrank(x: _SampleData, y: _SampleData, alternative: Alternative = "two-sided") -> LogRankResult: ...
