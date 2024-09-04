# TODO: Finish this

import numpy.typing as npt
from scipy._typing import Untyped, UntypedCallable

EPS: Untyped

def validate_first_step(first_step: Untyped, t0: Untyped, t_bound: Untyped) -> Untyped: ...
def validate_max_step(max_step: Untyped) -> Untyped: ...
def warn_extraneous(extraneous: Untyped) -> Untyped: ...
def validate_tol(rtol: Untyped, atol: Untyped, n: int) -> tuple[Untyped, Untyped]: ...
def norm(x: npt.ArrayLike) -> Untyped: ...
def select_initial_step(
    fun: UntypedCallable,
    t0: Untyped,
    y0: Untyped,
    t_bound: Untyped,
    max_step: Untyped,
    f0: Untyped,
    direction: Untyped,
    order: Untyped,
    rtol: Untyped,
    atol: Untyped,
) -> Untyped: ...

class OdeSolution:
    n_segments: Untyped
    ts: Untyped
    interpolants: Untyped
    t_min: Untyped
    t_max: Untyped
    ascending: bool
    side: Untyped
    ts_sorted: Untyped
    def __init__(self, ts: Untyped, interpolants: Untyped, alt_segment: bool = False) -> None: ...
    def __call__(self, t: Untyped) -> Untyped: ...

NUM_JAC_DIFF_REJECT: Untyped
NUM_JAC_DIFF_SMALL: Untyped
NUM_JAC_DIFF_BIG: Untyped
NUM_JAC_MIN_FACTOR: Untyped
NUM_JAC_FACTOR_INCREASE: int
NUM_JAC_FACTOR_DECREASE: float

def num_jac(
    fun: UntypedCallable,
    t: Untyped,
    y: Untyped,
    f: Untyped,
    threshold: Untyped,
    factor: Untyped,
    sparsity: Untyped | None = None,
) -> Untyped: ...
