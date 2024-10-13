from typing import Generic, Literal, NamedTuple, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import Alternative
from ._resampling import PermutationMethod

_FloatOrArray: TypeAlias = float | np.float64 | npt.NDArray[np.float64]
_FloatOrArrayT = TypeVar("_FloatOrArrayT", bound=_FloatOrArray, default=_FloatOrArray)

class MannwhitneyuResult(NamedTuple, Generic[_FloatOrArrayT]):
    statistic: _FloatOrArrayT
    pvalue: _FloatOrArrayT

def mannwhitneyu(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    use_continuity: bool = True,
    alternative: Alternative = "two-sided",
    axis: int = 0,
    method: Literal["auto", "asymptotic", "exact"] | PermutationMethod = "auto",
    *,
    nan_policy: Literal["propagate", "raise", "coerce", "omit"] = "propagate",
    keepdims: bool = False,
) -> MannwhitneyuResult: ...
