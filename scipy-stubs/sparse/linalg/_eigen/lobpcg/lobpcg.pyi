from scipy._typing import Untyped

__all__ = ["lobpcg"]

def lobpcg(
    A: Untyped,
    X: Untyped,
    B: Untyped | None = None,
    M: Untyped | None = None,
    Y: Untyped | None = None,
    tol: Untyped | None = None,
    maxiter: Untyped | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: bool = False,
    retResidualNormsHistory: bool = False,
    restartControl: int = 20,
) -> Untyped: ...
