from typing import Any, Literal, TypeAlias
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
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

def _find_repeats(arr: npt.NDArray[np.number[Any]]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...
def siegelslopes(
    y: npt.ArrayLike,
    x: npt.ArrayLike | None = None,
    method: _Method = "hierarchical",
) -> SiegelslopesResult: ...
def theilslopes(
    y: npt.ArrayLike,
    x: npt.ArrayLike | None = None,
    alpha: float | np.floating[Any] = 0.95,
    method: _Method = "separate",
) -> TheilslopesResult: ...
