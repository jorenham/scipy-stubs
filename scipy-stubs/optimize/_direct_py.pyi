from collections.abc import Callable, Iterable
from typing import Any

import numpy.typing as npt
from scipy._typing import Untyped
from ._constraints import Bounds
from ._optimize import OptimizeResult

__all__ = ["direct"]

ERROR_MESSAGES: tuple[str, ...]
SUCCESS_MESSAGES: tuple[str, ...]

def direct(
    func: Callable[[npt.ArrayLike, tuple[Any]], float],
    bounds: Iterable[float] | Bounds,
    *,
    args: tuple[Untyped, ...] = (),
    eps: float = 0.0001,
    maxfun: int | None = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = ...,
    f_min_rtol: float = 0.0001,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-06,
    callback: Callable[[npt.ArrayLike], None] | None = None,
) -> OptimizeResult: ...
