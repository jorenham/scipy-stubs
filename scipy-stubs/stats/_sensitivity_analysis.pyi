from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, type_check_only

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG
from ._resampling import BootstrapResult

__all__ = ["sobol_indices"]

_SobolKey: TypeAlias = Literal["f_A", "f_B", "f_AB"]
_SobolMethod: TypeAlias = Callable[
    [onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
    tuple[onp.ToFloat | onp.ToFloatND, onp.ToFloat | onp.ToFloatND],
]

@type_check_only
class _HasPPF(Protocol):
    @property
    def ppf(self, /) -> Callable[..., np.float64]: ...

###

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
def f_ishigami(x: onp.ToFloat2D) -> onp.Array1D[np.floating[Any]]: ...

#
def sample_A_B(n: onp.ToInt, dists: Sequence[_HasPPF], rng: ToRNG = None) -> onp.ArrayND[np.float64]: ...
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
    func: Callable[[onp.Array2D[np.float64]], onp.ToComplex2D] | Mapping[_SobolKey, onp.ArrayND[np.number[Any]]],
    n: onp.ToJustInt,
    dists: Sequence[_HasPPF] | None = None,
    method: _SobolMethod | Literal["saltelli_2010"] = "saltelli_2010",
    rng: ToRNG = None,
) -> SobolResult: ...
