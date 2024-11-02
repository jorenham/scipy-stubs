from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyInt, AnyReal, Seed
from ._resampling import BootstrapResult

__all__ = ["sobol_indices"]

_SobolKey: TypeAlias = Literal["f_A", "f_B", "f_AB"]
_SobolMethod: TypeAlias = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.ArrayLike, npt.ArrayLike],
]

###

# this protocol exists at runtime (but it has an incorrect reutrn type, which is corrected here)
class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., np.float64]: ...

@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: npt.NDArray[np.float64]
    total_order: npt.NDArray[np.float64]
    _indices_method: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    _f_A: npt.NDArray[np.float64]
    _f_B: npt.NDArray[np.float64]
    _f_AB: npt.NDArray[np.float64]
    _A: npt.NDArray[np.float64] | None = None
    _B: npt.NDArray[np.float64] | None = None
    _AB: npt.NDArray[np.float64] | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(self, confidence_level: AnyReal = 0.95, n_resamples: AnyInt = 999) -> BootstrapSobolResult: ...

#
def f_ishigami(x: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]: ...

#
def sample_A_B(n: AnyInt, dists: Sequence[PPFDist], random_state: Seed | None = None) -> npt.NDArray[np.float64]: ...
def sample_AB(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

#
def saltelli_2010(
    f_A: npt.NDArray[np.float64],
    f_B: npt.NDArray[np.float64],
    f_AB: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

#
def sobol_indices(
    *,
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike] | Mapping[_SobolKey, npt.NDArray[np.number[Any]]],
    n: AnyInt,
    dists: Sequence[PPFDist] | None = None,
    method: _SobolMethod | Literal["saltelli_2010"] = "saltelli_2010",
    random_state: Seed | None = None,
) -> SobolResult: ...
