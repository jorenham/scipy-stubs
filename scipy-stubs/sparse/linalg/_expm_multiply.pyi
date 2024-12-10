from scipy._typing import Untyped

__all__ = ["expm_multiply"]

class LazyOperatorNormInfo:
    def __init__(self, /, A: Untyped, A_1_norm: Untyped | None = None, ell: int = 2, scale: int = 1) -> None: ...
    def set_scale(self, /, scale: Untyped) -> None: ...
    def onenorm(self, /) -> Untyped: ...
    def d(self, /, p: Untyped) -> Untyped: ...
    def alpha(self, /, p: Untyped) -> Untyped: ...

#
def traceest(A: Untyped, m3: Untyped, seed: Untyped | None = None) -> Untyped: ...

#
def expm_multiply(
    A: Untyped,
    B: Untyped,
    start: Untyped | None = None,
    stop: Untyped | None = None,
    num: Untyped | None = None,
    endpoint: Untyped | None = None,
    traceA: Untyped | None = None,
) -> Untyped: ...
