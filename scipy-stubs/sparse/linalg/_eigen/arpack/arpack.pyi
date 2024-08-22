from scipy._lib._threadsafety import ReentrancyLock as ReentrancyLock
from scipy._typing import Untyped
from scipy.linalg import eig as eig, eigh as eigh, lu_factor as lu_factor, lu_solve as lu_solve
from scipy.sparse import eye as eye, issparse as issparse
from scipy.sparse._sputils import (
    convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy,
    is_pydata_spmatrix as is_pydata_spmatrix,
    isdense as isdense,
)
from scipy.sparse.linalg import gmres as gmres, splu as splu
from scipy.sparse.linalg._interface import LinearOperator as LinearOperator, aslinearoperator as aslinearoperator

arpack_int: Untyped
__docformat__: str
DNAUPD_ERRORS: Untyped
SNAUPD_ERRORS = DNAUPD_ERRORS
ZNAUPD_ERRORS: Untyped
CNAUPD_ERRORS = ZNAUPD_ERRORS
DSAUPD_ERRORS: Untyped
SSAUPD_ERRORS = DSAUPD_ERRORS
DNEUPD_ERRORS: Untyped
SNEUPD_ERRORS: Untyped
ZNEUPD_ERRORS: Untyped
CNEUPD_ERRORS: Untyped
DSEUPD_ERRORS: Untyped
SSEUPD_ERRORS: Untyped

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
    def iterate(self): ...
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
    def iterate(self): ...
    def extract(self, return_eigenvectors) -> Untyped: ...

class SpLuInv(LinearOperator):
    M_lu: Untyped
    shape: Untyped
    dtype: Untyped
    isreal: Untyped
    def __init__(self, M) -> None: ...

class LuInv(LinearOperator):
    M_lu: Untyped
    shape: Untyped
    dtype: Untyped
    def __init__(self, M) -> None: ...

def gmres_loose(A, b, tol) -> Untyped: ...

class IterInv(LinearOperator):
    M: Untyped
    dtype: Untyped
    shape: Untyped
    ifunc: Untyped
    tol: Untyped
    def __init__(self, M, ifunc=..., tol: int = 0): ...

class IterOpInv(LinearOperator):
    A: Untyped
    M: Untyped
    sigma: Untyped
    OP: Untyped
    shape: Untyped
    ifunc: Untyped
    tol: Untyped
    def __init__(self, A, M, sigma, ifunc=..., tol: int = 0): ...
    @property
    def dtype(self) -> Untyped: ...

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
