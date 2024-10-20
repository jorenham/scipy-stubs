from typing import Final
from typing_extensions import override

import numpy as np
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
    def __init__(self, info: Untyped, infodict: Untyped = ...) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Untyped
    eigenvectors: Untyped
    def __init__(self, msg: Untyped, eigenvalues: Untyped, eigenvectors: Untyped) -> None: ...

def choose_ncv(k: Untyped) -> Untyped: ...

class SpLuInv(LinearOperator):
    M_lu: Untyped
    isreal: Untyped
    def __init__(self, M: Untyped) -> None: ...

class LuInv(LinearOperator):
    M_lu: Untyped
    def __init__(self, M: Untyped) -> None: ...

def gmres_loose(A: Untyped, b: Untyped, tol: Untyped) -> Untyped: ...

class IterInv(LinearOperator):
    M: Untyped
    ifunc: Untyped
    tol: Untyped
    def __init__(self, M: Untyped, ifunc: Untyped = ..., tol: float = 0) -> None: ...

class IterOpInv(LinearOperator):
    A: Untyped
    M: Untyped
    sigma: Untyped
    OP: Untyped
    ifunc: Untyped
    tol: Untyped
    @property
    @override
    def dtype(self, /) -> np.dtype[np.generic]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleVariableOverride]
    def __init__(self, A: Untyped, M: Untyped, sigma: Untyped, ifunc: Untyped = ..., tol: float = 0) -> None: ...

def get_inv_matvec(M: Untyped, hermitian: bool = False, tol: float = 0) -> Untyped: ...
def get_OPinv_matvec(A: Untyped, M: Untyped, sigma: Untyped, hermitian: bool = False, tol: float = 0) -> Untyped: ...
def eigs(
    A: Untyped,
    k: int = 6,
    M: Untyped | None = None,
    sigma: Untyped | None = None,
    which: str = "LM",
    v0: Untyped | None = None,
    ncv: Untyped | None = None,
    maxiter: Untyped | None = None,
    tol: float = 0,
    return_eigenvectors: bool = True,
    Minv: Untyped | None = None,
    OPinv: Untyped | None = None,
    OPpart: Untyped | None = None,
) -> Untyped: ...
def eigsh(
    A: Untyped,
    k: int = 6,
    M: Untyped | None = None,
    sigma: Untyped | None = None,
    which: str = "LM",
    v0: Untyped | None = None,
    ncv: Untyped | None = None,
    maxiter: Untyped | None = None,
    tol: float = 0,
    return_eigenvectors: bool = True,
    Minv: Untyped | None = None,
    OPinv: Untyped | None = None,
    mode: str = "normal",
) -> Untyped: ...
