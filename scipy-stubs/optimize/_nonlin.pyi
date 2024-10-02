from typing_extensions import override

import numpy as np
from scipy._typing import Untyped, UntypedCallable

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

def maxnorm(x: Untyped) -> Untyped: ...
def nonlin_solve(
    F: Untyped,
    x0: Untyped,
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
        norm: Untyped = ...,
    ) -> None: ...
    def check(self, f: Untyped, x: Untyped, dx: Untyped) -> Untyped: ...

class Jacobian:
    func: Untyped
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    def __init__(self, **kw: Untyped) -> None: ...
    def aspreconditioner(self) -> Untyped: ...
    def solve(self, v: Untyped, tol: float = 0) -> Untyped: ...
    def update(self, x: Untyped, F: Untyped) -> None: ...
    def setup(self, x: Untyped, F: Untyped, func: Untyped) -> None: ...

class InverseJacobian:
    jacobian: Untyped
    matvec: UntypedCallable
    rmatvec: UntypedCallable
    update: UntypedCallable
    setup: UntypedCallable
    def __init__(self, jacobian: Untyped) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[np.generic]: ...

def asjacobian(J: Untyped) -> Untyped: ...

class GenericBroyden(Jacobian):
    last_f: Untyped
    last_x: Untyped
    alpha: Untyped
    @override
    def update(self, x0: Untyped, f0: Untyped) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setup(self, x0: Untyped, f0: Untyped, func: Untyped) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class LowRankMatrix:
    alpha: Untyped
    cs: Untyped
    ds: Untyped
    n: Untyped
    dtype: Untyped
    collapsed: Untyped
    def __init__(self, alpha: Untyped, n: Untyped, dtype: Untyped) -> None: ...
    def matvec(self, v: Untyped) -> Untyped: ...
    def rmatvec(self, v: Untyped) -> Untyped: ...
    def solve(self, v: Untyped, tol: float = 0) -> Untyped: ...
    def rsolve(self, v: Untyped, tol: float = 0) -> Untyped: ...
    def append(self, c: Untyped, d: Untyped) -> None: ...
    def __array__(self, dtype: Untyped | None = None, copy: Untyped | None = None) -> Untyped: ...
    def collapse(self) -> None: ...
    def restart_reduce(self, rank: Untyped) -> None: ...
    def simple_reduce(self, rank: Untyped) -> None: ...
    def svd_reduce(self, max_rank: Untyped, to_retain: Untyped | None = None) -> None: ...

class BroydenFirst(GenericBroyden):
    alpha: Untyped
    Gm: Untyped
    max_rank: Untyped
    def __init__(
        self,
        alpha: Untyped | None = None,
        reduction_method: str = "restart",
        max_rank: Untyped | None = None,
    ) -> None: ...
    def todense(self) -> Untyped: ...
    @override
    def solve(self, f: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, f: Untyped) -> Untyped: ...
    def rsolve(self, f: Untyped, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f: Untyped) -> Untyped: ...

class BroydenSecond(BroydenFirst): ...

class Anderson(GenericBroyden):
    alpha: Untyped
    M: Untyped
    dx: Untyped
    df: Untyped
    gamma: Untyped
    w0: Untyped
    def __init__(self, alpha: Untyped | None = None, w0: float = 0.01, M: int = 5) -> None: ...
    @override
    def solve(self, f: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, f: Untyped) -> Untyped: ...

class DiagBroyden(GenericBroyden):
    alpha: Untyped
    d: Untyped
    def __init__(self, alpha: Untyped | None = None) -> None: ...
    @override
    def solve(self, f: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, f: Untyped) -> Untyped: ...
    def rsolve(self, f: Untyped, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f: Untyped) -> Untyped: ...
    def todense(self) -> Untyped: ...

class LinearMixing(GenericBroyden):
    alpha: Untyped
    def __init__(self, alpha: Untyped | None = None) -> None: ...
    @override
    def solve(self, f: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, f: Untyped) -> Untyped: ...
    def rsolve(self, f: Untyped, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f: Untyped) -> Untyped: ...
    def todense(self) -> Untyped: ...

class ExcitingMixing(GenericBroyden):
    alpha: Untyped
    alphamax: Untyped
    beta: Untyped
    def __init__(self, alpha: Untyped | None = None, alphamax: float = 1.0) -> None: ...
    @override
    def solve(self, f: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, f: Untyped) -> Untyped: ...
    def rsolve(self, f: Untyped, tol: float = 0) -> Untyped: ...
    def rmatvec(self, f: Untyped) -> Untyped: ...
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
        **kw: Untyped,
    ) -> None: ...
    def matvec(self, v: Untyped) -> Untyped: ...
    @override
    def solve(self, rhs: Untyped, tol: float = 0) -> Untyped: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, x: Untyped, f: Untyped) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setup(self, x: Untyped, f: Untyped, func: Untyped) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

broyden1: Untyped
broyden2: Untyped
anderson: Untyped
linearmixing: Untyped
diagbroyden: Untyped
excitingmixing: Untyped
newton_krylov: Untyped
