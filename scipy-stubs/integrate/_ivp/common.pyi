from scipy._typing import Untyped
from scipy.sparse import coo_matrix as coo_matrix, find as find

EPS: Untyped

def validate_first_step(first_step, t0, t_bound) -> Untyped: ...
def validate_max_step(max_step) -> Untyped: ...
def warn_extraneous(extraneous): ...
def validate_tol(rtol, atol, n) -> Untyped: ...
def norm(x) -> Untyped: ...
def select_initial_step(fun, t0, y0, t_bound, max_step, f0, direction, order, rtol, atol) -> Untyped: ...

class OdeSolution:
    n_segments: Untyped
    ts: Untyped
    interpolants: Untyped
    t_min: Untyped
    t_max: Untyped
    ascending: bool
    side: Untyped
    ts_sorted: Untyped
    def __init__(self, ts, interpolants, alt_segment: bool = False): ...
    def __call__(self, t) -> Untyped: ...

NUM_JAC_DIFF_REJECT: Untyped
NUM_JAC_DIFF_SMALL: Untyped
NUM_JAC_DIFF_BIG: Untyped
NUM_JAC_MIN_FACTOR: Untyped
NUM_JAC_FACTOR_INCREASE: int
NUM_JAC_FACTOR_DECREASE: float

def num_jac(fun, t, y, f, threshold, factor, sparsity: Untyped | None = None) -> Untyped: ...
