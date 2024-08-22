from scipy._lib._array_api import array_namespace as array_namespace
from scipy._typing import Untyped

def differentiate(
    f,
    x,
    *,
    args=(),
    tolerances: Untyped | None = None,
    maxiter: int = 10,
    order: int = 8,
    initial_step: float = 0.5,
    step_factor: float = 2.0,
    step_direction: int = 0,
    preserve_shape: bool = False,
    callback: Untyped | None = None,
) -> Untyped: ...
def jacobian(
    f,
    x,
    *,
    tolerances: Untyped | None = None,
    maxiter: int = 10,
    order: int = 8,
    initial_step: float = 0.5,
    step_factor: float = 2.0,
) -> Untyped: ...
