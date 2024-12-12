from typing import Any, Literal, TypeAlias
from typing_extensions import Self

import numpy as np
import optype.numpy as onp
from . import distributions as distributions
from ._typing import BaseBunch

__all__ = ["_find_repeats", "siegelslopes", "theilslopes"]

_Method: TypeAlias = Literal["hierarchical", "separate"]

class SiegelslopesResult(BaseBunch[float, float]):
    def __new__(_cls, slope: float, intercept: float) -> Self: ...
    def __init__(self, /, slope: float, intercept: float) -> None: ...
    @property
    def slope(self, /) -> float: ...
    @property
    def intercept(self, /) -> float: ...

class TheilslopesResult(BaseBunch[np.float64, np.float64, np.float64, np.float64]):
    def __new__(_cls, slope: np.float64, intercept: np.float64, low_slope: np.float64, high_slope: np.float64) -> Self: ...
    def __init__(self, /, slope: np.float64, intercept: np.float64, low_slope: np.float64, high_slope: np.float64) -> None: ...
    @property
    def slope(self, /) -> np.float64: ...
    @property
    def intercept(self, /) -> np.float64: ...
    @property
    def low_slope(self, /) -> np.float64: ...
    @property
    def high_slope(self, /) -> np.float64: ...

def _find_repeats(arr: onp.ArrayND[np.number[Any]]) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...
def siegelslopes(
    y: onp.ToFloatND,
    x: onp.ToFloat1D | None = None,
    method: _Method = "hierarchical",
) -> SiegelslopesResult: ...
def theilslopes(
    y: onp.ToFloatND,
    x: onp.ToFloat1D | None = None,
    alpha: float | np.floating[Any] = 0.95,
    method: _Method = "separate",
) -> TheilslopesResult: ...
