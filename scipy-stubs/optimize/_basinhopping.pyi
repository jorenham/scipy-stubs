from scipy._typing import Untyped

__all__ = ["basinhopping"]

def basinhopping(
    func: Untyped,
    x0: Untyped,
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
