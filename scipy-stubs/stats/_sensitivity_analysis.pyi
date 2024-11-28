from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy._typing import Seed
from ._resampling import BootstrapResult

__all__ = ["sobol_indices"]

_SobolKey: TypeAlias = Literal["f_A", "f_B", "f_AB"]
_SobolMethod: TypeAlias = Callable[
    [onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
    tuple[npt.ArrayLike, npt.ArrayLike],
]

###

# this protocol exists at runtime (but it has an incorrect reutrn type, which is corrected here)
class PPFDist(Protocol):
    @property
    def ppf(self, /) -> Callable[..., np.float64]: ...

@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: onp.ArrayND[np.float64]
    total_order: onp.ArrayND[np.float64]
    _indices_method: Callable[[onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
    _f_A: onp.ArrayND[np.float64]
    _f_B: onp.ArrayND[np.float64]
    _f_AB: onp.ArrayND[np.float64]
    _A: onp.ArrayND[np.float64] | None = None
    _B: onp.ArrayND[np.float64] | None = None
    _AB: onp.ArrayND[np.float64] | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(self, /, confidence_level: onp.ToFloat = 0.95, n_resamples: onp.ToInt = 999) -> BootstrapSobolResult: ...

#
def f_ishigami(x: npt.ArrayLike) -> onp.ArrayND[np.floating[Any]]: ...

#
def sample_A_B(n: onp.ToInt, dists: Sequence[PPFDist], random_state: Seed | None = None) -> onp.ArrayND[np.float64]: ...
def sample_AB(A: onp.ArrayND[np.float64], B: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...

#
def saltelli_2010(
    f_A: onp.ArrayND[np.float64],
    f_B: onp.ArrayND[np.float64],
    f_AB: onp.ArrayND[np.float64],
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...

#
def sobol_indices(
    *,
    func: Callable[[onp.ArrayND[np.float64]], npt.ArrayLike] | Mapping[_SobolKey, onp.ArrayND[np.number[Any]]],
    n: onp.ToInt,
    dists: Sequence[PPFDist] | None = None,
    method: _SobolMethod | Literal["saltelli_2010"] = "saltelli_2010",
    random_state: Seed | None = None,
) -> SobolResult: ...
