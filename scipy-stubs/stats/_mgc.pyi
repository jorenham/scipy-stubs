from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypedDict, type_check_only
from typing_extensions import Self, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy._typing import Seed
from ._typing import BaseBunch

_T = TypeVar("_T")
_R = TypeVar("_R")

@type_check_only
class _MGCDict(TypedDict):
    mgc_map: onpt.Array[tuple[int, int], np.float64]
    opt_scale: Sequence[int | np.intp]  # list of size 2
    null_dist: onpt.Array[tuple[int], np.float64]

class MGCResult(BaseBunch[np.float64, np.float64, _MGCDict]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def pvalue(self, /) -> np.float64: ...
    @property
    def mgc_dict(self, /) -> _MGCDict: ...
    def __new__(_cls, statistic: np.float64, pvalue: np.float64, mgc_dict: _MGCDict) -> Self: ...
    def __init__(self, /, statistic: np.float64, pvalue: np.float64, mgc_dict: _MGCDict) -> None: ...

def multiscale_graphcorr(
    x: npt.NDArray[np.floating[Any] | np.integer[Any] | np.bool_],
    y: npt.NDArray[np.floating[Any] | np.integer[Any] | np.bool_],
    compute_distance: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.floating[Any]]] = ...,
    reps: int = 1000,
    workers: int | Callable[[Callable[[_T], _R], Iterable[_T]], Sequence[_R]] = 1,
    is_twosamp: bool = False,
    random_state: Seed | None = None,
) -> MGCResult: ...
