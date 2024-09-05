from typing import ClassVar

from scipy._typing import Untyped, UntypedArray, UntypedCallable
from .base import DenseOutput, OdeSolver

SAFETY: float
MIN_FACTOR: float
MAX_FACTOR: int

def rk_step(
    fun: UntypedCallable,
    t: Untyped,
    y: Untyped,
    f: Untyped,
    h: Untyped,
    A: Untyped,
    B: Untyped,
    C: Untyped,
    K: Untyped,
) -> Untyped: ...

class RungeKutta(OdeSolver):
    C: ClassVar[UntypedArray]
    A: ClassVar[UntypedArray]
    B: ClassVar[UntypedArray]
    E: ClassVar[UntypedArray]
    P: ClassVar[UntypedArray]
    order: ClassVar[int]
    error_estimator_order: ClassVar[int]
    n_stages: ClassVar[int]
    y_old: Untyped
    max_step: Untyped
    f: Untyped
    h_abs: Untyped
    K: Untyped
    error_exponent: Untyped
    h_previous: Untyped
    def __init__(
        self,
        /,
        fun: UntypedCallable,
        t0: Untyped,
        y0: Untyped,
        t_bound: Untyped,
        max_step: Untyped = ...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous: Untyped,
    ) -> None: ...

class RK23(RungeKutta): ...
class RK45(RungeKutta): ...

class DOP853(RungeKutta):
    E3: ClassVar[UntypedArray]
    E5: ClassVar[UntypedArray]
    D: ClassVar[UntypedArray]
    A_EXTRA: ClassVar[UntypedArray]
    C_EXTRA: ClassVar[UntypedArray]

    K_extended: Untyped
    K: Untyped
    def __init__(
        self,
        /,
        fun: UntypedCallable,
        t0: Untyped,
        y0: Untyped,
        t_bound: Untyped,
        max_step: Untyped = ...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous: Untyped,
    ) -> None: ...

class RkDenseOutput(DenseOutput):
    h: Untyped
    Q: Untyped
    order: Untyped
    y_old: Untyped
    def __init__(self, /, t_old: float, t: float, y_old: Untyped, Q: Untyped) -> None: ...

class Dop853DenseOutput(DenseOutput):
    h: Untyped
    F: Untyped
    y_old: Untyped
    def __init__(self, /, t_old: float, t: float, y_old: Untyped, F: Untyped) -> None: ...
