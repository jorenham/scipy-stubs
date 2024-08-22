from scipy._lib._array_api import (
    array_namespace as array_namespace,
    xp_atleast_nd as xp_atleast_nd,
    xp_copy as xp_copy,
    xp_cov as xp_cov,
    xp_size as xp_size,
)
from scipy._lib._util import check_random_state as check_random_state, rng_integers as rng_integers
from scipy._typing import Untyped
from scipy.spatial.distance import cdist as cdist

__docformat__: str

class ClusterError(Exception): ...

def whiten(obs, check_finite: bool = True) -> Untyped: ...
def vq(obs, code_book, check_finite: bool = True) -> Untyped: ...
def py_vq(obs, code_book, check_finite: bool = True) -> Untyped: ...
def kmeans(
    obs, k_or_guess, iter: int = 20, thresh: float = 1e-05, check_finite: bool = True, *, seed: Untyped | None = None
) -> Untyped: ...
def kmeans2(
    data,
    k,
    iter: int = 10,
    thresh: float = 1e-05,
    minit: str = "random",
    missing: str = "warn",
    check_finite: bool = True,
    *,
    seed: Untyped | None = None,
) -> Untyped: ...
