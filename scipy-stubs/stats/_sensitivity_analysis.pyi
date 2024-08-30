from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy._typing as spt
from scipy.stats._resampling import BootstrapResult as BootstrapResult

__all__ = ["sobol_indices"]

def f_ishigami(x: npt.ArrayLike) -> spt.UntypedArray: ...
def sample_A_B(
    n: spt.AnyInt,
    dists: list[PPFDist],
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> spt.UntypedArray: ...
def sample_AB(A: spt.UntypedArray, B: spt.UntypedArray) -> spt.UntypedArray: ...
def saltelli_2010(
    f_A: spt.UntypedArray,
    f_B: spt.UntypedArray,
    f_AB: spt.UntypedArray,
) -> tuple[spt.UntypedArray, spt.UntypedArray]: ...
@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: spt.UntypedArray
    total_order: spt.UntypedArray
    def bootstrap(self, confidence_level: spt.AnyReal = 0.95, n_resamples: spt.AnyInt = 999) -> BootstrapSobolResult: ...

class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., float]: ...

_SobolKey: TypeAlias = Literal["f_A", "f_B", "f_AB"]

def sobol_indices(
    *,
    func: Callable[[npt.NDArray[np.number[Any]]], npt.ArrayLike] | dict[_SobolKey, npt.NDArray[np.number[Any]]],
    n: spt.AnyInt,
    dists: list[PPFDist] | None = None,
    method: Callable[..., spt.Untyped] | Literal["saltelli_2010"] = "saltelli_2010",
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> SobolResult: ...
