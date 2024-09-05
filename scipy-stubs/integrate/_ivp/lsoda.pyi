# TODO: Finish this

from scipy._typing import Untyped, UntypedCallable
from .base import DenseOutput, OdeSolver

class LSODA(OdeSolver):
    def __init__(
        self,
        /,
        fun: UntypedCallable,
        t0: Untyped,
        y0: Untyped,
        t_bound: Untyped,
        first_step: Untyped | None = None,
        min_step: float = 0.0,
        max_step: Untyped = ...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        jac: Untyped | None = None,
        lband: Untyped | None = None,
        uband: Untyped | None = None,
        vectorized: bool = False,
        **extraneous: Untyped,
    ) -> None: ...

class LsodaDenseOutput(DenseOutput):
    h: Untyped
    yh: Untyped
    p: Untyped
    def __init__(self, /, t_old: float, t: float, h: Untyped, order: Untyped, yh: Untyped) -> None: ...
