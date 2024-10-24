import abc
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Final, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeInt
from optype import CanBool, CanFloat, CanIndex, CanInt, CanLen
from scipy._typing import RNG, AnyBool, AnyInt, AnyReal, Seed
from scipy.spatial.distance import _MetricCallback, _MetricKind

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
_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)
_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])
_ArrayT_f = TypeVar("_ArrayT_f", bound=npt.NDArray[np.floating[Any]])
_N = TypeVar("_N", bound=int)

@type_check_only
class _CanLenArray(CanLen, onpt.CanArray[Any, np.dtype[_SCT_co]], Protocol[_SCT_co]): ...

_Scalar_f_co: TypeAlias = np.floating[Any] | np.integer[Any] | np.bool_

_Array1D: TypeAlias = onpt.Array[tuple[int], _SCT]
_Array1D_f8: TypeAlias = _Array1D[np.float64]
_Array2D: TypeAlias = onpt.Array[tuple[int, int], _SCT]
_Array2D_f8: TypeAlias = _Array2D[np.float64]

_Any1D_f: TypeAlias = _CanLenArray[np.floating[Any]] | Sequence[float | np.floating[Any]]
_Any1D_f_co: TypeAlias = _CanLenArray[_Scalar_f_co] | Sequence[AnyReal]
_Any2D_f: TypeAlias = _CanLenArray[np.floating[Any]] | Sequence[Sequence[float | np.floating[Any]]] | Sequence[_Any1D_f]
_Any2D_f_co: TypeAlias = _CanLenArray[_Scalar_f_co] | Sequence[Sequence[AnyReal]] | Sequence[_Any1D_f_co]

_MethodOptimize: TypeAlias = Literal["random-cd", "lloyd"]
_MethodDisc: TypeAlias = Literal["CD", "WD", "MD", "L2-star"]
_MethodDist: TypeAlias = Literal["mindist", "mst"]
_MetricDist: TypeAlias = _MetricKind | _MetricCallback

_FuncOptimize: TypeAlias = Callable[[_Any2D_f], _Any2D_f]

###

class QMCEngine(abc.ABC):
    d: Final[int | np.integer[Any]]
    rng_seed: Final[np.random.Generator]
    optimization_method: Final[_FuncOptimize | None]
    rng: np.random.Generator
    num_generated: int

    @abc.abstractmethod
    def __init__(self, /, d: AnyInt, *, optimization: _MethodOptimize | None = None, seed: Seed | None = None) -> None: ...
    def random(self, /, n: AnyInt = 1, *, workers: AnyInt = 1) -> _Array2D_f8: ...
    def integers(
        self,
        /,
        l_bounds: _ArrayLikeInt,
        *,
        u_bounds: _ArrayLikeInt | None = None,
        n: AnyInt = 1,
        endpoint: AnyBool = False,
        workers: AnyInt = 1,
    ) -> _Array2D[np.int64]: ...
    def reset(self, /) -> Self: ...
    def fast_forward(self, /, n: AnyInt) -> Self: ...

class Halton(QMCEngine):
    base: list[int]
    scramble: bool

    def __init__(
        self,
        /,
        d: AnyInt,
        *,
        scramble: bool = True,
        optimization: _MethodOptimize | None = None,
        seed: Seed | None = None,
    ) -> None: ...

class LatinHypercube(QMCEngine):
    scramble: bool
    lhs_method: Callable[[int | np.integer[Any]], _Array2D_f8]

    def __init__(
        self,
        /,
        d: AnyInt,
        *,
        scramble: bool = True,
        strength: int = 1,
        optimization: _MethodOptimize | None = None,
        seed: Seed | None = None,
    ) -> None: ...

class Sobol(QMCEngine):
    MAXDIM: ClassVar[int] = 21_201

    dtype_i: Final[type[np.uint32 | np.uint64]]
    bits: int | np.integer[Any]
    maxn: int | np.integer[Any]

    def __init__(
        self,
        /,
        d: AnyInt,
        *,
        scramble: CanBool = True,
        bits: AnyInt | None = None,
        seed: Seed | None = None,
        optimization: _MethodOptimize | None = None,
    ) -> None: ...
    def random_base2(self, /, m: AnyInt) -> _Array2D_f8: ...

@type_check_only
class _HypersphereMethod(Protocol):
    def __call__(
        self,
        /,
        center: npt.NDArray[_Scalar_f_co],
        radius: AnyReal,
        candidates: AnyInt = 1,
    ) -> _Array2D_f8: ...

class PoissonDisk(QMCEngine):
    hypersphere_method: Final[_HypersphereMethod]
    radius_factor: Final[float]
    radius: Final[AnyReal]
    radius_squared: Final[AnyReal]
    ncandidates: Final[AnyInt]
    cell_size: Final[np.float64]
    grid_size: Final[_Array1D[np.int_]]

    sample_pool: list[_Array1D_f8]
    sample_grid: npt.NDArray[np.float32]

    def __init__(
        self,
        /,
        d: AnyInt,
        *,
        radius: AnyReal = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: AnyInt = 30,
        optimization: _MethodOptimize | None = None,
        seed: Seed | None = None,
    ) -> None: ...
    def fill_space(self, /) -> _Array2D_f8: ...

class MultivariateNormalQMC:
    engine: Final[QMCEngine]
    def __init__(
        self,
        /,
        mean: _Any1D_f_co,
        cov: _Any2D_f_co | None = None,
        *,
        cov_root: _Any2D_f_co | None = None,
        inv_transform: CanBool = True,
        engine: QMCEngine | None = None,
        seed: Seed | None = None,
    ) -> None: ...
    def random(self, /, n: AnyInt = 1) -> _Array2D_f8: ...

class MultinomialQMC:
    pvals: Final[_Array1D[np.floating[Any]]]
    n_trials: Final[AnyInt]
    engine: Final[QMCEngine]

    def __init__(
        self,
        /,
        pvals: _Any1D_f | float | np.floating[Any],
        n_trials: AnyInt,
        *,
        engine: QMCEngine | None = None,
        seed: Seed | None = None,
    ) -> None: ...
    def random(self, /, n: AnyInt = 1) -> _Array2D_f8: ...

#
@overload
def check_random_state(seed: int | np.integer[Any] | numbers.Integral | None = None) -> np.random.Generator: ...
@overload
def check_random_state(seed: _RNGT) -> _RNGT: ...
def scale(
    sample: _Any2D_f,
    l_bounds: _Any1D_f_co | AnyReal,
    u_bounds: _Any1D_f_co | AnyReal,
    *,
    reverse: CanBool = False,
) -> _Array2D_f8: ...
def discrepancy(sample: _Any2D_f, *, iterative: CanBool = False, method: _MethodDisc = "CD", workers: CanInt = 1) -> float: ...
def geometric_discrepancy(sample: _Any2D_f, method: _MethodDist = "mindist", metric: _MetricDist = "euclidean") -> np.float64: ...
def update_discrepancy(x_new: _Any1D_f, sample: _Any2D_f, initial_disc: CanFloat) -> float: ...
def primes_from_2_to(n: AnyInt) -> _Array1D[np.int_]: ...
def n_primes(n: AnyInt) -> list[int] | _Array1D[np.int_]: ...

#
def _select_optimizer(optimization: _MethodOptimize | None, config: Mapping[str, object]) -> _FuncOptimize | None: ...
def _random_cd(best_sample: _ArrayT_f, n_iters: AnyInt, n_nochange: AnyInt, rng: RNG) -> _ArrayT_f: ...
def _l1_norm(sample: _Any2D_f) -> np.float64: ...
def _lloyd_iteration(sample: _ArrayT_f, decay: AnyReal, qhull_options: str | None) -> _ArrayT_f: ...
def _lloyd_centroidal_voronoi_tessellation(
    sample: _Any2D_f,
    *,
    tol: AnyReal = 1e-5,
    maxiter: AnyInt = 10,
    qhull_options: str | None = None,
) -> _Array2D_f8: ...

#
def _ensure_in_unit_hypercube(sample: _Any2D_f) -> _Array2D_f8: ...
@overload
def _perturb_discrepancy(
    sample: _Array2D[np.integer[Any] | np.bool_],
    i1: CanIndex,
    i2: CanIndex,
    k: CanIndex,
    disc: AnyReal,
) -> np.float64: ...
@overload
def _perturb_discrepancy(
    sample: _Array2D[_SCT_fc],
    i1: CanIndex,
    i2: CanIndex,
    k: CanIndex,
    disc: AnyReal,
) -> _SCT_fc: ...
@overload
def _van_der_corput_permutation(base: CanIndex, *, random_state: Seed | None = None) -> _Array2D[np.int_]: ...
@overload
def _van_der_corput_permutation(base: CanFloat, *, random_state: Seed | None = None) -> _Array2D_f8: ...
def van_der_corput(
    n: CanInt,
    base: AnyInt = 2,
    *,
    start_index: AnyInt = 0,
    scramble: CanBool = False,
    permutations: _ArrayLikeInt | None = None,
    seed: Seed | None = None,
    workers: CanInt = 1,
) -> _Array1D_f8: ...

#
@overload
def _validate_workers(workers: CanInt[Literal[1]] | CanIndex[Literal[1]] | Literal[1] = 1) -> Literal[1]: ...
@overload
def _validate_workers(workers: _N) -> _N: ...
@overload
def _validate_workers(workers: CanInt[_N] | CanIndex[_N]) -> _N: ...
def _validate_bounds(
    l_bounds: _Any1D_f_co,
    u_bounds: _Any1D_f_co,
    d: AnyInt,
) -> tuple[_Array1D[_Scalar_f_co], _Array1D[_Scalar_f_co]]: ...
