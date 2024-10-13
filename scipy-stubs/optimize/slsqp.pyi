# This file is not meant for public use and will be removed in SciPy v2.0.0.

from collections.abc import Callable
from typing_extensions import Any, deprecated

__all__ = ["OptimizeResult", "fmin_slsqp", "slsqp", "zeros"]

@deprecated("will be removed in SciPy v2.0.0")
def zeros(
    shape: object,
    dtype: object = ...,
    order: object = ...,
    *,
    device: object = ...,
    like: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
class OptimizeResult(Any): ...

@deprecated("will be removed in SciPy v2.0.0")
def fmin_slsqp(
    func: object,
    x0: object,
    eqcons: object = ...,
    f_eqcons: object = ...,
    ieqcons: object = ...,
    f_ieqcons: object = ...,
    bounds: object = ...,
    fprime: object = ...,
    fprime_eqcons: object = ...,
    fprime_ieqcons: object = ...,
    args: object = ...,
    iter: object = ...,
    acc: object = ...,
    iprint: object = ...,
    disp: object = ...,
    full_output: object = ...,
    epsilon: object = ...,
    callback: object = ...,
) -> object: ...

# Deprecated
slsqp: Callable[..., object]
