from .base import DenseOutput as DenseOutput, OdeSolver as OdeSolver
from .common import validate_first_step as validate_first_step, validate_tol as validate_tol, warn_extraneous as warn_extraneous
from scipy._typing import Untyped
from scipy.integrate import ode as ode

class LSODA(OdeSolver):
    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        first_step: Untyped | None = None,
        min_step: float = 0.0,
        max_step=...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        jac: Untyped | None = None,
        lband: Untyped | None = None,
        uband: Untyped | None = None,
        vectorized: bool = False,
        **extraneous,
    ): ...

class LsodaDenseOutput(DenseOutput):
    h: Untyped
    yh: Untyped
    p: Untyped
    def __init__(self, t_old, t, h, order, yh) -> None: ...
