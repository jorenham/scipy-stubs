from typing import ClassVar, Literal

import numpy as np
from scipy._typing import Untyped, UntypedArray
from .base import DenseOutput, OdeSolver

SAFETY: float
MIN_FACTOR: float
MAX_FACTOR: int

def rk_step(fun, t, y, f, h, A, B, C, K) -> Untyped: ...

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
        fun,
        t0,
        y0,
        t_bound,
        max_step=...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous,
    ): ...

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
        fun,
        t0,
        y0,
        t_bound,
        max_step=...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        vectorized: bool = False,
        first_step: Untyped | None = None,
        **extraneous,
    ): ...

class RkDenseOutput(DenseOutput):
    h: Untyped
    Q: Untyped
    order: Untyped
    y_old: Untyped
    def __init__(self, t_old, t, y_old, Q) -> None: ...

class Dop853DenseOutput(DenseOutput):
    h: Untyped
    F: Untyped
    y_old: Untyped
    def __init__(self, t_old, t, y_old, F) -> None: ...
