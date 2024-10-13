from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy._typing as spt
from ._resampling import BootstrapResult

__all__ = ["sobol_indices"]

_SobolKey: TypeAlias = Literal["f_A", "f_B", "f_AB"]

# exists at runtime
class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., float]: ...

@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: spt.UntypedArray
    total_order: spt.UntypedArray
    _indices_method: spt.UntypedCallable
    _f_A: spt.UntypedArray
    _f_B: spt.UntypedArray
    _f_AB: spt.UntypedArray
    _A: spt.UntypedArray | None = None
    _B: spt.UntypedArray | None = None
    _AB: spt.UntypedArray | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(self, confidence_level: spt.AnyReal = 0.95, n_resamples: spt.AnyInt = 999) -> BootstrapSobolResult: ...

def f_ishigami(x: npt.ArrayLike) -> spt.UntypedArray: ...
def sample_A_B(
    n: spt.AnyInt,
    dists: list[PPFDist],
    random_state: spt.Seed | None = None,
) -> spt.UntypedArray: ...
def sample_AB(A: spt.UntypedArray, B: spt.UntypedArray) -> spt.UntypedArray: ...
def saltelli_2010(
    f_A: spt.UntypedArray,
    f_B: spt.UntypedArray,
    f_AB: spt.UntypedArray,
) -> tuple[spt.UntypedArray, spt.UntypedArray]: ...
def sobol_indices(
    *,
    func: Callable[[npt.NDArray[np.number[Any]]], npt.ArrayLike] | dict[_SobolKey, npt.NDArray[np.number[Any]]],
    n: spt.AnyInt,
    dists: list[PPFDist] | None = None,
    method: Callable[..., spt.Untyped] | Literal["saltelli_2010"] = "saltelli_2010",
    random_state: spt.Seed | None = None,
) -> SobolResult: ...
