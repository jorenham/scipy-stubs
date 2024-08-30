from typing import NamedTuple

from scipy import optimize as optimize, stats as stats
from scipy._lib._util import check_random_state as check_random_state
from scipy._typing import Untyped

class FitResult:
    discrete: Untyped
    pxf: Untyped
    params: Untyped
    success: Untyped
    message: Untyped
    def __init__(self, dist, data, discrete, res) -> None: ...
    def nllf(self, params: Untyped | None = None, data: Untyped | None = None) -> Untyped: ...
    def plot(self, ax: Untyped | None = None, *, plot_type: str = "hist") -> Untyped: ...

def fit(
    dist, data, bounds: Untyped | None = None, *, guess: Untyped | None = None, method: str = "mle", optimizer=...
) -> Untyped: ...

class GoodnessOfFitResult(NamedTuple):
    fit_result: Untyped
    statistic: Untyped
    pvalue: Untyped
    null_distribution: Untyped

def goodness_of_fit(
    dist,
    data,
    *,
    known_params: Untyped | None = None,
    fit_params: Untyped | None = None,
    guessed_params: Untyped | None = None,
    statistic: str = "ad",
    n_mc_samples: int = 9999,
    random_state: Untyped | None = None,
) -> Untyped: ...
