from abc import ABC, abstractmethod
from typing import ClassVar, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy._lib._util import DecimalNumber, IntNumber, SeedType
from scipy._typing import Untyped, UntypedArray

__all__ = [
    "Halton",
    "LatinHypercube",
    "MultinomialQMC",
    "MultivariateNormalQMC",
    "PoissonDisk",
    "QMCEngine",
    "Sobol",
    "discrepancy",
    "geometric_discrepancy",
    "scale",
    "update_discrepancy",
]

_RNGT = TypeVar("_RNGT", bound=np.random.Generator | np.random.RandomState)

@overload
def check_random_state(seed: IntNumber | None = ...) -> np.random.Generator: ...
@overload
def check_random_state(seed: _RNGT) -> _RNGT: ...
def scale(sample: npt.ArrayLike, l_bounds: npt.ArrayLike, u_bounds: npt.ArrayLike, *, reverse: bool = False) -> UntypedArray: ...
def discrepancy(
    sample: npt.ArrayLike,
    *,
    iterative: bool = False,
    method: Literal["CD", "WD", "MD", "L2-star"] = "CD",
    workers: IntNumber = 1,
) -> float: ...
def geometric_discrepancy(
    sample: npt.ArrayLike,
    method: Literal["mindist", "mst"] = "mindist",
    metric: str = "euclidean",
) -> float: ...
def update_discrepancy(x_new: npt.ArrayLike, sample: npt.ArrayLike, initial_disc: DecimalNumber) -> float: ...
def primes_from_2_to(n: int) -> UntypedArray: ...
def n_primes(n: IntNumber) -> list[int]: ...
def van_der_corput(
    n: IntNumber,
    base: IntNumber = 2,
    *,
    start_index: IntNumber = 0,
    scramble: bool = False,
    permutations: npt.ArrayLike | None = None,
    seed: SeedType = None,
    workers: IntNumber = 1,
) -> UntypedArray: ...

class QMCEngine(ABC):
    d: Untyped
    rng: Untyped
    rng_seed: Untyped
    num_generated: int
    optimization_method: Untyped
    @abstractmethod
    def __init__(
        self,
        d: IntNumber,
        *,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None,
    ) -> None: ...
    def random(self, n: IntNumber = 1, *, workers: IntNumber = 1) -> UntypedArray: ...
    def integers(
        self,
        l_bounds: npt.ArrayLike,
        *,
        u_bounds: npt.ArrayLike | None = None,
        n: IntNumber = 1,
        endpoint: bool = False,
        workers: IntNumber = 1,
    ) -> UntypedArray: ...
    def reset(self) -> QMCEngine: ...
    def fast_forward(self, n: IntNumber) -> QMCEngine: ...

class Halton(QMCEngine):
    seed: Untyped
    base: Untyped
    scramble: Untyped
    def __init__(
        self,
        d: IntNumber,
        *,
        scramble: bool = True,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None,
    ): ...

class LatinHypercube(QMCEngine):
    scramble: Untyped
    lhs_method: Untyped
    def __init__(
        self,
        d: IntNumber,
        *,
        scramble: bool = True,
        strength: int = 1,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None,
    ): ...

class Sobol(QMCEngine):
    MAXDIM: ClassVar[int]
    bits: Untyped
    dtype_i: Untyped
    maxn: Untyped
    def __init__(
        self,
        d: IntNumber,
        *,
        scramble: bool = True,
        bits: IntNumber | None = None,
        seed: SeedType = None,
        optimization: Literal["random-cd", "lloyd"] | None = None,
    ): ...
    def random_base2(self, m: IntNumber) -> UntypedArray: ...
    def reset(self) -> Sobol: ...
    def fast_forward(self, n: IntNumber) -> Sobol: ...

class PoissonDisk(QMCEngine):
    hypersphere_method: Untyped
    radius_factor: Untyped
    radius: Untyped
    radius_squared: Untyped
    ncandidates: Untyped
    cell_size: Untyped
    grid_size: Untyped
    def __init__(
        self,
        d: IntNumber,
        *,
        radius: DecimalNumber = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: IntNumber = 30,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None,
        l_bounds: npt.ArrayLike | None = None,
        u_bounds: npt.ArrayLike | None = None,
    ): ...
    def fill_space(self) -> UntypedArray: ...
    def reset(self) -> PoissonDisk: ...

class MultivariateNormalQMC:
    engine: Untyped
    def __init__(
        self,
        mean: npt.ArrayLike,
        cov: npt.ArrayLike | None = None,
        *,
        cov_root: npt.ArrayLike | None = None,
        inv_transform: bool = True,
        engine: QMCEngine | None = None,
        seed: SeedType = None,
    ): ...
    def random(self, n: IntNumber = 1) -> UntypedArray: ...

class MultinomialQMC:
    pvals: Untyped
    n_trials: Untyped
    engine: Untyped
    def __init__(self, pvals: npt.ArrayLike, n_trials: IntNumber, *, engine: QMCEngine | None = None, seed: SeedType = None): ...
    def random(self, n: IntNumber = 1) -> UntypedArray: ...
