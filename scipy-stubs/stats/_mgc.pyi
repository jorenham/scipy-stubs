from scipy._lib._util import MapWrapper as MapWrapper, check_random_state as check_random_state, rng_integers as rng_integers
from scipy._typing import Untyped
from scipy.spatial.distance import cdist as cdist
from . import distributions as distributions

class _ParallelP:
    x: Untyped
    y: Untyped
    random_states: Untyped
    def __init__(self, x, y, random_states) -> None: ...
    def __call__(self, index) -> Untyped: ...

MGCResult: Untyped

def multiscale_graphcorr(
    x, y, compute_distance=..., reps: int = 1000, workers: int = 1, is_twosamp: bool = False, random_state: Untyped | None = None
) -> Untyped: ...
