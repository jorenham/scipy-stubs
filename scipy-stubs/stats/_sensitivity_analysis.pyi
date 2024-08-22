from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from scipy._lib._util import DecimalNumber as DecimalNumber, IntNumber as IntNumber, SeedType as SeedType
from scipy.stats import bootstrap as bootstrap, qmc as qmc
from scipy.stats._common import ConfidenceInterval as ConfidenceInterval
from scipy.stats._qmc import check_random_state as check_random_state
from scipy.stats._resampling import BootstrapResult as BootstrapResult

def f_ishigami(x: npt.ArrayLike) -> np.ndarray: ...
def sample_A_B(n: IntNumber, dists: list[PPFDist], random_state: SeedType = None) -> np.ndarray: ...
def sample_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray: ...
def saltelli_2010(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

@dataclass
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    def bootstrap(self, confidence_level: DecimalNumber = 0.95, n_resamples: IntNumber = 999) -> BootstrapSobolResult: ...

class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., float]: ...

def sobol_indices(
    *,
    func: Callable[[np.ndarray], npt.ArrayLike] | dict[Literal["f_A", "f_B", "f_AB"], np.ndarray],
    n: IntNumber,
    dists: list[PPFDist] | None = None,
    method: Callable | Literal["saltelli_2010"] = "saltelli_2010",
    random_state: SeedType = None,
) -> SobolResult: ...
