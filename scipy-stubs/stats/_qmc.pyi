import abc
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Final, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import Self, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import RNG, Seed
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
_SCT0 = TypeVar("_SCT0", bound=np.generic, default=np.float64)
_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)
_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])
_ArrayT_f = TypeVar("_ArrayT_f", bound=onp.ArrayND[np.floating[Any]])
_N = TypeVar("_N", bound=int)

# the `__len__` ensures that scalar types like `np.generic` are excluded
@type_check_only
class _CanLenArray(Protocol[_SCT_co]):
    def __len__(self, /) -> int: ...
    def __array__(self, /) -> onp.ArrayND[_SCT_co]: ...

_Scalar_f_co: TypeAlias = np.floating[Any] | np.integer[Any] | np.bool_
_ScalarLike_f: TypeAlias = float | np.floating[Any]

_Array1D: TypeAlias = onp.Array1D[_SCT0]
_Array2D: TypeAlias = onp.Array2D[_SCT0]
_Array1D_f_co: TypeAlias = _Array1D[_Scalar_f_co]

_Any1D_f: TypeAlias = _CanLenArray[np.floating[Any]] | Sequence[_ScalarLike_f]
_Any1D_f_co: TypeAlias = _CanLenArray[_Scalar_f_co] | Sequence[onp.ToFloat]
_Any2D_f: TypeAlias = _CanLenArray[np.floating[Any]] | Sequence[Sequence[_ScalarLike_f]] | Sequence[_Any1D_f]
_Any2D_f_co: TypeAlias = _CanLenArray[_Scalar_f_co] | Sequence[Sequence[onp.ToFloat]] | Sequence[_Any1D_f_co]

_MethodQMC: TypeAlias = Literal["random-cd", "lloyd"]
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
    def __init__(self, /, d: onp.ToInt, *, optimization: _MethodQMC | None = None, seed: Seed | None = None) -> None: ...
    def random(self, /, n: opt.AnyInt = 1, *, workers: onp.ToInt = 1) -> _Array2D: ...
    def integers(
        self,
        /,
        l_bounds: onp.ToInt | onp.ToIntND,
        *,
        u_bounds: onp.ToInt | onp.ToIntND | None = None,
        n: opt.AnyInt = 1,
        endpoint: op.CanBool = False,
        workers: opt.AnyInt = 1,
    ) -> _Array2D[np.int64]: ...
    def reset(self, /) -> Self: ...
    def fast_forward(self, /, n: opt.AnyInt) -> Self: ...

class Halton(QMCEngine):
    base: list[int]
    scramble: bool

    def __init__(
        self,
        /,
        d: onp.ToInt,
        *,
        scramble: bool = True,
        optimization: _MethodQMC | None = None,
        seed: Seed | None = None,
    ) -> None: ...

class LatinHypercube(QMCEngine):
    scramble: bool
    lhs_method: Callable[[int | np.integer[Any]], _Array2D]

    def __init__(
        self,
        /,
        d: onp.ToInt,
        *,
        scramble: bool = True,
        strength: int = 1,
        optimization: _MethodQMC | None = None,
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
        d: onp.ToInt,
        *,
        scramble: op.CanBool = True,
        bits: onp.ToInt | None = None,
        optimization: _MethodQMC | None = None,
        seed: Seed | None = None,
    ) -> None: ...
    def random_base2(self, /, m: onp.ToInt) -> _Array2D: ...

@type_check_only
class _HypersphereMethod(Protocol):
    def __call__(
        self,
        /,
        center: onp.ArrayND[_Scalar_f_co],
        radius: onp.ToFloat,
        candidates: onp.ToInt = 1,
    ) -> _Array2D: ...

class PoissonDisk(QMCEngine):
    hypersphere_method: Final[_HypersphereMethod]
    radius_factor: Final[float]
    radius: Final[onp.ToFloat]
    radius_squared: Final[onp.ToFloat]
    ncandidates: Final[onp.ToInt]
    cell_size: Final[np.float64]
    grid_size: Final[_Array1D[np.int_]]

    sample_pool: list[_Array1D]
    sample_grid: onp.ArrayND[np.float32]

    def __init__(
        self,
        /,
        d: onp.ToInt,
        *,
        radius: onp.ToFloat = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: onp.ToInt = 30,
        optimization: _MethodQMC | None = None,
        seed: Seed | None = None,
    ) -> None: ...
    def fill_space(self, /) -> _Array2D: ...

@type_check_only
class _QMCDistribution:
    engine: Final[QMCEngine]  # defaults to `Sobol`
    def __init__(self, /, *, engine: QMCEngine | None = None, seed: Seed | None = None) -> None: ...
    def random(self, /, n: onp.ToInt = 1) -> _Array2D: ...

class MultivariateNormalQMC(_QMCDistribution):
    @override
    def __init__(
        self,
        /,
        mean: _Any1D_f_co,
        cov: _Any2D_f_co | None = None,
        *,
        cov_root: _Any2D_f_co | None = None,
        inv_transform: op.CanBool = True,
        engine: QMCEngine | None = None,
        seed: Seed | None = None,
    ) -> None: ...

class MultinomialQMC(_QMCDistribution):
    pvals: Final[_Array1D[np.floating[Any]]]
    n_trials: Final[onp.ToInt]

    @override
    def __init__(
        self,
        /,
        pvals: _Any1D_f | float | np.floating[Any],
        n_trials: onp.ToInt,
        *,
        engine: QMCEngine | None = None,
        seed: Seed | None = None,
    ) -> None: ...

#
@overload
def check_random_state(seed: int | np.integer[Any] | numbers.Integral | None = None) -> np.random.Generator: ...
@overload
def check_random_state(seed: _RNGT) -> _RNGT: ...

#
def scale(
    sample: _Any2D_f,
    l_bounds: _Any1D_f_co | onp.ToFloat,
    u_bounds: _Any1D_f_co | onp.ToFloat,
    *,
    reverse: op.CanBool = False,
) -> _Array2D: ...

#
def discrepancy(
    sample: _Any2D_f,
    *,
    iterative: op.CanBool = False,
    method: _MethodDisc = "CD",
    workers: opt.AnyInt = 1,
) -> float | np.float64: ...

#
def geometric_discrepancy(
    sample: _Any2D_f,
    method: _MethodDist = "mindist",
    metric: _MetricDist = "euclidean",
) -> float | np.float64: ...
def update_discrepancy(x_new: _Any1D_f, sample: _Any2D_f, initial_disc: opt.AnyFloat) -> float: ...
def primes_from_2_to(n: onp.ToInt) -> _Array1D[np.int_]: ...
def n_primes(n: onp.ToInt) -> list[int] | _Array1D[np.int_]: ...

#
def _select_optimizer(optimization: _MethodQMC | None, config: Mapping[str, object]) -> _FuncOptimize | None: ...
def _random_cd(best_sample: _ArrayT_f, n_iters: onp.ToInt, n_nochange: onp.ToInt, rng: RNG) -> _ArrayT_f: ...
def _l1_norm(sample: _Any2D_f) -> float | np.float64: ...
def _lloyd_iteration(sample: _ArrayT_f, decay: onp.ToFloat, qhull_options: str | None) -> _ArrayT_f: ...
def _lloyd_centroidal_voronoi_tessellation(
    sample: _Any2D_f,
    *,
    tol: onp.ToFloat = 1e-5,
    maxiter: onp.ToInt = 10,
    qhull_options: str | None = None,
) -> _Array2D: ...
def _ensure_in_unit_hypercube(sample: _Any2D_f) -> _Array2D: ...

#
@overload
def _perturb_discrepancy(
    sample: _Array2D[np.integer[Any] | np.bool_],
    i1: op.CanIndex,
    i2: op.CanIndex,
    k: op.CanIndex,
    disc: onp.ToFloat,
) -> float | np.float64: ...
@overload
def _perturb_discrepancy(
    sample: _Array2D[_SCT_fc],
    i1: op.CanIndex,
    i2: op.CanIndex,
    k: op.CanIndex,
    disc: onp.ToFloat,
) -> _SCT_fc: ...

#
@overload
def _van_der_corput_permutation(base: op.CanIndex, *, random_state: Seed | None = None) -> _Array2D[np.int_]: ...
@overload
def _van_der_corput_permutation(base: op.CanFloat, *, random_state: Seed | None = None) -> _Array2D: ...

#
def van_der_corput(
    n: op.CanInt,
    base: onp.ToInt = 2,
    *,
    start_index: onp.ToInt = 0,
    scramble: op.CanBool = False,
    permutations: onp.ToInt | onp.ToIntND | None = None,
    seed: Seed | None = None,
    workers: opt.AnyInt = 1,
) -> _Array1D: ...

#
@overload
def _validate_workers(workers: Literal[1] = 1) -> Literal[1]: ...
@overload
def _validate_workers(workers: _N) -> _N: ...
@overload
def _validate_workers(workers: opt.AnyInt[_N]) -> _N: ...

#
def _validate_bounds(l_bounds: _Any1D_f_co, u_bounds: _Any1D_f_co, d: onp.ToInt) -> tuple[_Array1D_f_co, _Array1D_f_co]: ...
