from typing import Final

from scipy._typing import Untyped
from scipy.sparse.linalg._interface import LinearOperator

__all__ = ["ArpackError", "ArpackNoConvergence", "eigs", "eigsh"]

DNAUPD_ERRORS: Final[dict[int, str]]
SNAUPD_ERRORS = DNAUPD_ERRORS
ZNAUPD_ERRORS: Final[dict[int, str]]
CNAUPD_ERRORS = ZNAUPD_ERRORS
DSAUPD_ERRORS: Final[dict[int, str]]
SSAUPD_ERRORS = DSAUPD_ERRORS
DNEUPD_ERRORS: Final[dict[int, str]]
SNEUPD_ERRORS: Final[dict[int, str]]
ZNEUPD_ERRORS: Final[dict[int, str]]
CNEUPD_ERRORS: Final[dict[int, str]]
DSEUPD_ERRORS: Final[dict[int, str]]
SSEUPD_ERRORS: Final[dict[int, str]]

class ArpackError(RuntimeError):
    def __init__(self, info, infodict=...) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Untyped
    eigenvectors: Untyped
    def __init__(self, msg, eigenvalues, eigenvectors) -> None: ...

def choose_ncv(k) -> Untyped: ...

class _ArpackParams:
    resid: Untyped
    sigma: int
    v: Untyped
    iparam: Untyped
    mode: Untyped
    n: Untyped
    tol: Untyped
    k: Untyped
    maxiter: Untyped
    ncv: Untyped
    which: Untyped
    tp: Untyped
    info: Untyped
    converged: bool
    ido: int
    def __init__(
        self,
        n,
        k,
        tp,
        mode: int = 1,
        sigma: Untyped | None = None,
        ncv: Untyped | None = None,
        v0: Untyped | None = None,
        maxiter: Untyped | None = None,
        which: str = "LM",
        tol: int = 0,
    ): ...

class _SymmetricArpackParams(_ArpackParams):
    OP: Untyped
    B: Untyped
    bmat: str
    OPa: Untyped
    OPb: Untyped
    A_matvec: Untyped
    workd: Untyped
    workl: Untyped
    iterate_infodict: Untyped
    extract_infodict: Untyped
    ipntr: Untyped
    def __init__(
        self,
        n,
        k,
        tp,
        matvec,
        mode: int = 1,
        M_matvec: Untyped | None = None,
        Minv_matvec: Untyped | None = None,
        sigma: Untyped | None = None,
        ncv: Untyped | None = None,
        v0: Untyped | None = None,
        maxiter: Untyped | None = None,
        which: str = "LM",
        tol: int = 0,
    ): ...
    converged: bool
    def extract(self, return_eigenvectors) -> Untyped: ...

class _UnsymmetricArpackParams(_ArpackParams):
    OP: Untyped
    B: Untyped
    bmat: str
    OPa: Untyped
    OPb: Untyped
    matvec: Untyped
    workd: Untyped
    workl: Untyped
    iterate_infodict: Untyped
    extract_infodict: Untyped
    ipntr: Untyped
    rwork: Untyped
    def __init__(
        self,
        n,
        k,
        tp,
        matvec,
        mode: int = 1,
        M_matvec: Untyped | None = None,
        Minv_matvec: Untyped | None = None,
        sigma: Untyped | None = None,
        ncv: Untyped | None = None,
        v0: Untyped | None = None,
        maxiter: Untyped | None = None,
        which: str = "LM",
        tol: int = 0,
    ): ...
    converged: bool
    def extract(self, return_eigenvectors) -> Untyped: ...

class SpLuInv(LinearOperator):
    M_lu: Untyped
    isreal: Untyped
    def __init__(self, M) -> None: ...

class LuInv(LinearOperator):
    M_lu: Untyped
    def __init__(self, M) -> None: ...

def gmres_loose(A, b, tol) -> Untyped: ...

class IterInv(LinearOperator):
    M: Untyped
    ifunc: Untyped
    tol: Untyped
    def __init__(self, M, ifunc=..., tol: int = 0): ...

class IterOpInv(LinearOperator):
    A: Untyped
    M: Untyped
    sigma: Untyped
    OP: Untyped
    ifunc: Untyped
    tol: Untyped
    def __init__(self, A, M, sigma, ifunc=..., tol: int = 0): ...

def get_inv_matvec(M, hermitian: bool = False, tol: int = 0) -> Untyped: ...
def get_OPinv_matvec(A, M, sigma, hermitian: bool = False, tol: int = 0) -> Untyped: ...
def eigs(
    A,
    k: int = 6,
    M: Untyped | None = None,
    sigma: Untyped | None = None,
    which: str = "LM",
    v0: Untyped | None = None,
    ncv: Untyped | None = None,
    maxiter: Untyped | None = None,
    tol: int = 0,
    return_eigenvectors: bool = True,
    Minv: Untyped | None = None,
    OPinv: Untyped | None = None,
    OPpart: Untyped | None = None,
) -> Untyped: ...
def eigsh(
    A,
    k: int = 6,
    M: Untyped | None = None,
    sigma: Untyped | None = None,
    which: str = "LM",
    v0: Untyped | None = None,
    ncv: Untyped | None = None,
    maxiter: Untyped | None = None,
    tol: int = 0,
    return_eigenvectors: bool = True,
    Minv: Untyped | None = None,
    OPinv: Untyped | None = None,
    mode: str = "normal",
) -> Untyped: ...
