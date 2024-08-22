from scipy._typing import Untyped
from scipy.linalg import (
    LinAlgError as LinAlgError,
    cho_factor as cho_factor,
    cho_solve as cho_solve,
    cholesky as cholesky,
    eigh as eigh,
    inv as inv,
)
from scipy.sparse import issparse as issparse
from scipy.sparse.linalg import LinearOperator as LinearOperator

def lobpcg(
    A,
    X,
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
