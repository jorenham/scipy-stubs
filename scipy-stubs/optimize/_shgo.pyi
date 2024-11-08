from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy._typing import EnterSelfMixin, Untyped, UntypedArray, UntypedCallable
from ._optimize import OptimizeResult as _OptimizeResult
from ._typing import Constraints

__all__ = ["shgo"]

_MinimizerKwargs: TypeAlias = Mapping[str, object]  # TODO(jorenham): TypedDict
_Options: TypeAlias = Mapping[str, object]  # TODO(jorenham): TypedDict

_SamplingMethodName: TypeAlias = Literal["simplicial", "halton", "sobol"]
_SamplingMethodFunc: TypeAlias = Callable[[int, int], npt.NDArray[np.float64]]
_SamplingMethod: TypeAlias = _SamplingMethodName | _SamplingMethodFunc

_VT = TypeVar("_VT")
_RT = TypeVar("_RT")

class OptimizeResult(_OptimizeResult):
    x: npt.NDArray[np.float64]
    xl: list[npt.NDArray[np.float64]]
    fun: float | np.float64
    funl: list[float | np.float64]
    success: bool
    message: str
    nfev: int
    nlfev: int
    nljev: int  # undocumented
    nlhev: int  # undocumented
    nit: int

###

class SHGO(EnterSelfMixin):
    func: UntypedCallable
    bounds: Untyped
    args: tuple[object, ...]
    callback: UntypedCallable
    dim: int
    constraints: Constraints
    min_cons: Untyped
    g_cons: Untyped
    g_args: Untyped
    minimizer_kwargs: _MinimizerKwargs
    f_min_true: Untyped
    minimize_every_iter: bool
    maxiter: int
    maxfev: int
    maxev: Untyped
    maxtime: Untyped
    minhgrd: Untyped
    symmetry: Untyped
    infty_cons_sampl: bool
    local_iter: bool
    disp: bool
    min_solver_args: Untyped
    stop_global: bool
    break_routine: bool
    iters: int
    iters_done: int
    n: int
    nc: int
    n_prc: int
    n_sampled: int
    fn: int
    hgr: int
    qhull_incremental: bool
    HC: Untyped
    iterate_complex: Untyped
    sampling_method: Untyped
    qmc_engine: Untyped
    sampling: Untyped
    sampling_function: Untyped
    stop_l_iter: bool
    stop_complex_iter: bool
    minimizer_pool: Untyped
    LMC: Untyped
    res: Untyped
    init: Untyped
    f_tol: Untyped
    f_lowest: Untyped
    x_lowest: Untyped
    hgrd: Untyped
    Ind_sorted: Untyped
    Tri: Untyped
    points: Untyped
    minimizer_pool_F: Untyped
    X_min: Untyped
    X_min_cache: Untyped
    Y: Untyped
    Z: Untyped
    Ss: Untyped
    ind_f_min: Untyped
    C: Untyped
    Xs: Untyped
    def __init__(
        self,
        func: UntypedCallable,
        bounds: Untyped,
        args: tuple[object, ...] = (),
        constraints: Constraints | None = None,
        n: Untyped | None = None,
        iters: Untyped | None = None,
        callback: UntypedCallable | None = None,
        minimizer_kwargs: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        sampling_method: _SamplingMethod = "simplicial",
        workers: int = 1,
    ) -> None: ...
    def init_options(self, options: Untyped) -> None: ...
    def iterate_all(self) -> None: ...
    def find_minima(self) -> None: ...
    def find_lowest_vertex(self) -> None: ...
    def finite_iterations(self) -> Untyped: ...
    def finite_fev(self) -> Untyped: ...
    def finite_ev(self) -> None: ...
    def finite_time(self) -> None: ...
    def finite_precision(self) -> Untyped: ...
    def finite_homology_growth(self) -> Untyped: ...
    def stopping_criteria(self) -> Untyped: ...
    def iterate(self) -> None: ...
    def iterate_hypercube(self) -> None: ...
    def iterate_delaunay(self) -> None: ...
    def minimizers(self) -> Untyped: ...
    def minimise_pool(self, force_iter: bool = False) -> None: ...
    def sort_min_pool(self) -> None: ...
    def trim_min_pool(self, trim_ind: Untyped) -> None: ...
    def g_topograph(self, x_min: Untyped, X_min: Untyped) -> Untyped: ...
    def construct_lcb_simplicial(self, v_min: Untyped) -> Untyped: ...
    def construct_lcb_delaunay(self, v_min: Untyped, ind: Untyped | None = None) -> Untyped: ...
    def minimize(self, x_min: Untyped, ind: Untyped | None = None) -> Untyped: ...
    def sort_result(self) -> Untyped: ...
    def fail_routine(self, mes: str = "Failed to converge") -> None: ...
    def sampled_surface(self, infty_cons_sampl: bool = False) -> None: ...
    def sampling_custom(self, n: Untyped, dim: Untyped) -> Untyped: ...
    def sampling_subspace(self) -> None: ...
    def sorted_samples(self) -> Untyped: ...
    def delaunay_triangulation(self, n_prc: int = 0) -> Untyped: ...

# undocumented
class LMap:
    v: Untyped
    x_l: Untyped
    lres: Untyped
    f_min: Untyped
    lbounds: list[Untyped]
    def __init__(self, /, v: Untyped) -> None: ...

# undocumented
class LMapCache:
    cache: Final[dict[tuple[np.number[Any], ...], Untyped]]
    xl_maps_set: Final[set[Untyped]]
    v_maps: Final[list[Untyped]]
    lbound_maps: Final[list[Untyped]]

    xl_maps: list[Untyped] | UntypedArray
    f_maps: list[Untyped]
    size: int

    def __init__(self, /) -> None: ...
    def __getitem__(self, v: npt.ArrayLike, /) -> Untyped: ...
    def add_res(self, /, v: Untyped, lres: Untyped, bounds: Untyped | None = None) -> None: ...
    def sort_cache_result(self, /) -> dict[str, Untyped]: ...  # TODO(jorenham): TypedDict

def shgo(
    func: UntypedCallable,
    bounds: Untyped,
    args: tuple[object, ...] = (),
    constraints: Constraints | None = None,
    n: int = 100,
    iters: int = 1,
    callback: Callable[[onpt.Array[tuple[int], np.float64]], None] | None = None,
    minimizer_kwargs: _MinimizerKwargs | None = None,  # TODO(jorenham): TypedDict
    options: _Options | None = None,  # TODO(jorenham): TypedDict
    sampling_method: _SamplingMethod = "simplicial",
    *,
    workers: int | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1,
) -> OptimizeResult: ...
