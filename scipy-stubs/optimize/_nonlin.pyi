from typing_extensions import override

from scipy._typing import Untyped

__all__ = [
    "BroydenFirst",
    "InverseJacobian",
    "KrylovJacobian",
    "NoConvergence",
    "anderson",
    "broyden1",
    "broyden2",
    "diagbroyden",
    "excitingmixing",
    "linearmixing",
    "newton_krylov",
]

class NoConvergence(Exception): ...

def maxnorm(x) -> Untyped: ...
def nonlin_solve(
    F,
    x0,
    jacobian: str = "krylov",
    iter: Untyped | None = None,
    verbose: bool = False,
    maxiter: Untyped | None = None,
    f_tol: Untyped | None = None,
    f_rtol: Untyped | None = None,
    x_tol: Untyped | None = None,
    x_rtol: Untyped | None = None,
    tol_norm: Untyped | None = None,
    line_search: str = "armijo",
    callback: Untyped | None = None,
    full_output: bool = False,
    raise_exception: bool = True,
) -> Untyped: ...

class TerminationCondition:
    x_tol: Untyped
    x_rtol: Untyped
    f_tol: Untyped
    f_rtol: Untyped
    norm: Untyped
    iter: Untyped
    f0_norm: Untyped
    iteration: int
    def __init__(
        self,
        f_tol: Untyped | None = None,
        f_rtol: Untyped | None = None,
        x_tol: Untyped | None = None,
        x_rtol: Untyped | None = None,
        iter: Untyped | None = None,
        norm=...,
    ) -> None: ...
    def check(self, f, x, dx) -> Untyped: ...

class Jacobian:
    func: Untyped
    @property
    def shape(self) -> Untyped: ...
    @property
    def dtype(self) -> Untyped: ...
    def __init__(self, **kw) -> None: ...
    def aspreconditioner(self) -> Untyped: ...
    def solve(self, v, /, tol: float = 0) -> Untyped: ...
    def update(self, x, F) -> None: ...
    def setup(self, x, F, func) -> None: ...

class InverseJacobian:
    jacobian: Untyped
    matvec: Untyped
    update: Untyped
    setup: Untyped
    rmatvec: Untyped
    def __init__(self, jacobian) -> None: ...
    @property
    def shape(self) -> Untyped: ...
    @property
    def dtype(self) -> Untyped: ...

def asjacobian(J) -> Untyped: ...

class GenericBroyden(Jacobian):
    last_f: Untyped
    last_x: Untyped
    alpha: Untyped

class LowRankMatrix:
    alpha: Untyped
    cs: Untyped
    ds: Untyped
    n: Untyped
    dtype: Untyped
    collapsed: Untyped
    def __init__(self, alpha, n, dtype) -> None: ...
    def matvec(self, v) -> Untyped: ...
    def rmatvec(self, v) -> Untyped: ...
    def solve(self, v, tol: float = 0) -> Untyped: ...
    def rsolve(self, v, tol: float = 0) -> Untyped: ...
    def append(self, c, d): ...
    def __array__(self, dtype: Untyped | None = None, copy: Untyped | None = None) -> Untyped: ...
    def collapse(self): ...
    def restart_reduce(self, rank): ...
    def simple_reduce(self, rank): ...
    def svd_reduce(self, max_rank, to_retain: Untyped | None = None): ...

class BroydenFirst(GenericBroyden):
    alpha: Untyped
    Gm: Untyped
    max_rank: Untyped
    def __init__(self, alpha: Untyped | None = None, reduction_method: str = "restart", max_rank: Untyped | None = None): ...
    def todense(self) -> Untyped: ...
    @override
    def solve(self, f, /, tol: float = 0) -> Untyped: ...
    def matvec(self, f) -> Untyped: ...
    def rsolve(self, f, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f) -> Untyped: ...

class BroydenSecond(BroydenFirst): ...

class Anderson(GenericBroyden):
    alpha: Untyped
    M: Untyped
    dx: Untyped
    df: Untyped
    gamma: Untyped
    w0: Untyped
    def __init__(self, alpha: Untyped | None = None, w0: float = 0.01, M: int = 5): ...
    @override
    def solve(self, f, tol: float = 0) -> Untyped: ...
    def matvec(self, f) -> Untyped: ...

class DiagBroyden(GenericBroyden):
    alpha: Untyped
    d: Untyped
    def __init__(self, alpha: Untyped | None = None) -> None: ...
    @override
    def solve(self, f, /, tol: float = 0) -> Untyped: ...
    def matvec(self, f) -> Untyped: ...
    def rsolve(self, f, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f) -> Untyped: ...
    def todense(self) -> Untyped: ...

class LinearMixing(GenericBroyden):
    alpha: Untyped
    def __init__(self, alpha: Untyped | None = None) -> None: ...
    @override
    def solve(self, f, /, tol: float = 0) -> Untyped: ...
    def matvec(self, f) -> Untyped: ...
    def rsolve(self, f, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f) -> Untyped: ...
    def todense(self) -> Untyped: ...

class ExcitingMixing(GenericBroyden):
    alpha: Untyped
    alphamax: Untyped
    beta: Untyped
    def __init__(self, alpha: Untyped | None = None, alphamax: float = 1.0) -> None: ...
    @override
    def solve(self, f, /, tol: float = 0) -> Untyped: ...
    def matvec(self, f) -> Untyped: ...
    def rsolve(self, f, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f) -> Untyped: ...
    def todense(self) -> Untyped: ...

class KrylovJacobian(Jacobian):
    preconditioner: Untyped
    rdiff: Untyped
    method: Untyped
    method_kw: Untyped
    x0: Untyped
    f0: Untyped
    op: Untyped
    def __init__(
        self,
        rdiff: Untyped | None = None,
        method: str = "lgmres",
        inner_maxiter: int = 20,
        inner_M: Untyped | None = None,
        outer_k: int = 10,
        **kw,
    ) -> None: ...
    def matvec(self, v) -> Untyped: ...
    @override
    def solve(self, rhs, /, tol: float = 0) -> Untyped: ...

broyden1: Untyped
broyden2: Untyped
anderson: Untyped
linearmixing: Untyped
diagbroyden: Untyped
excitingmixing: Untyped
newton_krylov: Untyped
