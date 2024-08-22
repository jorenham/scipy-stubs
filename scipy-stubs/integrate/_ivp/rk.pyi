import numpy as np

from . import dop853_coefficients as dop853_coefficients
from .base import DenseOutput as DenseOutput, OdeSolver as OdeSolver
from .common import (
    norm as norm,
    select_initial_step as select_initial_step,
    validate_first_step as validate_first_step,
    validate_max_step as validate_max_step,
    validate_tol as validate_tol,
    warn_extraneous as warn_extraneous,
)
from scipy._typing import Untyped

SAFETY: float
MIN_FACTOR: float
MAX_FACTOR: int

def rk_step(fun, t, y, f, h, A, B, C, K) -> Untyped: ...

class RungeKutta(OdeSolver):
    C: np.ndarray
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    P: np.ndarray
    order: int
    error_estimator_order: int
    n_stages: int
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

class RK23(RungeKutta):
    order: int
    error_estimator_order: int
    n_stages: int
    C: Untyped
    A: Untyped
    B: Untyped
    E: Untyped
    P: Untyped

class RK45(RungeKutta):
    order: int
    error_estimator_order: int
    n_stages: int
    C: Untyped
    A: Untyped
    B: Untyped
    E: Untyped
    P: Untyped

class DOP853(RungeKutta):
    n_stages: Untyped
    order: int
    error_estimator_order: int
    A: Untyped
    B: Untyped
    C: Untyped
    E3: Untyped
    E5: Untyped
    D: Untyped
    A_EXTRA: Untyped
    C_EXTRA: Untyped
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
