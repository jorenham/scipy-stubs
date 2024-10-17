from scipy import spatial as spatial
from scipy._typing import Untyped
from scipy.optimize import Bounds as Bounds, OptimizeResult as OptimizeResult, minimize as minimize
from scipy.optimize._constraints import new_bounds_to_old as new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints as standardize_constraints
from scipy.optimize._optimize import MemoizeJac as MemoizeJac
from scipy.optimize._shgo_lib._complex import Complex as Complex

def shgo(
    func: Untyped,
    bounds: Untyped,
    args: Untyped = (),
    constraints: Untyped | None = None,
    n: int = 100,
    iters: int = 1,
    callback: Untyped | None = None,
    minimizer_kwargs: Untyped | None = None,
    options: Untyped | None = None,
    sampling_method: str = "simplicial",
    *,
    workers: int = 1,
) -> Untyped: ...

class SHGO:
    func: Untyped
    bounds: Untyped
    args: Untyped
    callback: Untyped
    dim: Untyped
    constraints: Untyped
    min_cons: Untyped
    g_cons: Untyped
    g_args: Untyped
    minimizer_kwargs: Untyped
    f_min_true: Untyped
    minimize_every_iter: bool
    maxiter: Untyped
    maxfev: Untyped
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
    iters: Untyped
    iters_done: int
    n: Untyped
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
    def __init__(
        self,
        func: Untyped,
        bounds: Untyped,
        args: Untyped = (),
        constraints: Untyped | None = None,
        n: Untyped | None = None,
        iters: Untyped | None = None,
        callback: Untyped | None = None,
        minimizer_kwargs: Untyped | None = None,
        options: Untyped | None = None,
        sampling_method: str = "simplicial",
        workers: int = 1,
    ) -> None: ...
    init: Untyped
    f_tol: Untyped
    def init_options(self, options: Untyped) -> None: ...
    def __enter__(self) -> Untyped: ...
    def __exit__(self, *args: object) -> Untyped: ...
    def iterate_all(self) -> None: ...
    f_lowest: Untyped
    x_lowest: Untyped
    def find_minima(self) -> None: ...
    def find_lowest_vertex(self) -> None: ...
    def finite_iterations(self) -> Untyped: ...
    def finite_fev(self) -> Untyped: ...
    def finite_ev(self) -> None: ...
    def finite_time(self) -> None: ...
    def finite_precision(self) -> Untyped: ...
    hgrd: Untyped
    def finite_homology_growth(self) -> Untyped: ...
    def stopping_criteria(self) -> Untyped: ...
    def iterate(self) -> None: ...
    def iterate_hypercube(self) -> None: ...
    Ind_sorted: Untyped
    Tri: Untyped
    points: Untyped
    def iterate_delaunay(self) -> None: ...
    minimizer_pool_F: Untyped
    X_min: Untyped
    X_min_cache: Untyped
    def minimizers(self) -> Untyped: ...
    def minimise_pool(self, force_iter: bool = False) -> None: ...
    ind_f_min: Untyped
    def sort_min_pool(self) -> None: ...
    def trim_min_pool(self, trim_ind: Untyped) -> None: ...
    Y: Untyped
    Z: Untyped
    Ss: Untyped
    def g_topograph(self, x_min: Untyped, X_min: Untyped) -> Untyped: ...
    def construct_lcb_simplicial(self, v_min: Untyped) -> Untyped: ...
    def construct_lcb_delaunay(self, v_min: Untyped, ind: Untyped | None = None) -> Untyped: ...
    def minimize(self, x_min: Untyped, ind: Untyped | None = None) -> Untyped: ...
    def sort_result(self) -> Untyped: ...
    def fail_routine(self, mes: str = "Failed to converge") -> None: ...
    C: Untyped
    def sampled_surface(self, infty_cons_sampl: bool = False) -> None: ...
    def sampling_custom(self, n: Untyped, dim: Untyped) -> Untyped: ...
    def sampling_subspace(self) -> None: ...
    Xs: Untyped
    def sorted_samples(self) -> Untyped: ...
    def delaunay_triangulation(self, n_prc: int = 0) -> Untyped: ...

class LMap:
    v: Untyped
    x_l: Untyped
    lres: Untyped
    f_min: Untyped
    lbounds: Untyped
    def __init__(self, v: Untyped) -> None: ...

class LMapCache:
    cache: Untyped
    v_maps: Untyped
    xl_maps: Untyped
    xl_maps_set: Untyped
    f_maps: Untyped
    lbound_maps: Untyped
    size: int
    def __init__(self) -> None: ...
    def __getitem__(self, v: Untyped) -> Untyped: ...
    def add_res(self, v: Untyped, lres: Untyped, bounds: Untyped | None = None) -> None: ...
    def sort_cache_result(self) -> Untyped: ...
