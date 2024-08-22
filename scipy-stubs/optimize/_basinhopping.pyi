from scipy._lib._util import check_random_state as check_random_state
from scipy._typing import Untyped

class Storage:
    def __init__(self, minres) -> None: ...
    def update(self, minres) -> Untyped: ...
    def get_lowest(self) -> Untyped: ...

class BasinHoppingRunner:
    x: Untyped
    minimizer: Untyped
    step_taking: Untyped
    accept_tests: Untyped
    disp: Untyped
    nstep: int
    res: Untyped
    energy: Untyped
    incumbent_minres: Untyped
    storage: Untyped
    def __init__(self, x0, minimizer, step_taking, accept_tests, disp: bool = False): ...
    xtrial: Untyped
    energy_trial: Untyped
    accept: Untyped
    def one_cycle(self) -> Untyped: ...
    def print_report(self, energy_trial, accept): ...

class AdaptiveStepsize:
    takestep: Untyped
    target_accept_rate: Untyped
    interval: Untyped
    factor: Untyped
    verbose: Untyped
    nstep: int
    nstep_tot: int
    naccept: int
    def __init__(self, takestep, accept_rate: float = 0.5, interval: int = 50, factor: float = 0.9, verbose: bool = True): ...
    def __call__(self, x) -> Untyped: ...
    def take_step(self, x) -> Untyped: ...
    def report(self, accept, **kwargs): ...

class RandomDisplacement:
    stepsize: Untyped
    random_gen: Untyped
    def __init__(self, stepsize: float = 0.5, random_gen: Untyped | None = None): ...
    def __call__(self, x) -> Untyped: ...

class MinimizerWrapper:
    minimizer: Untyped
    func: Untyped
    kwargs: Untyped
    def __init__(self, minimizer, func: Untyped | None = None, **kwargs): ...
    def __call__(self, x0) -> Untyped: ...

class Metropolis:
    beta: Untyped
    random_gen: Untyped
    def __init__(self, T, random_gen: Untyped | None = None): ...
    def accept_reject(self, res_new, res_old) -> Untyped: ...
    def __call__(self, *, res_new, res_old) -> Untyped: ...

def basinhopping(
    func,
    x0,
    niter: int = 100,
    T: float = 1.0,
    stepsize: float = 0.5,
    minimizer_kwargs: Untyped | None = None,
    take_step: Untyped | None = None,
    accept_test: Untyped | None = None,
    callback: Untyped | None = None,
    interval: int = 50,
    disp: bool = False,
    niter_success: Untyped | None = None,
    seed: Untyped | None = None,
    *,
    target_accept_rate: float = 0.5,
    stepwise_factor: float = 0.9,
) -> Untyped: ...
