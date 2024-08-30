from scipy._lib._util import check_random_state as check_random_state
from scipy._typing import Untyped
from scipy.linalg import svd as svd
from scipy.sparse.linalg._eigen.lobpcg import lobpcg as lobpcg
from scipy.sparse.linalg._interface import LinearOperator as LinearOperator, aslinearoperator as aslinearoperator
from . import eigsh as eigsh

__all__ = ["svds"]

def svds(
    A,
    k: int = 6,
    ncv: Untyped | None = None,
    tol: int = 0,
    which: str = "LM",
    v0: Untyped | None = None,
    maxiter: Untyped | None = None,
    return_singular_vectors: bool = True,
    solver: str = "arpack",
    random_state: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
