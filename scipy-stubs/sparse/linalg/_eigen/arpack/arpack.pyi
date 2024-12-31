from collections.abc import Mapping
from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

__all__ = ["ArpackError", "ArpackNoConvergence", "eigs", "eigsh"]

_KT = TypeVar("_KT")

_ToRealMatrix: TypeAlias = onp.ToFloat2D | LinearOperator[np.floating[Any] | np.integer[Any]] | _spbase
_ToComplexMatrix: TypeAlias = onp.ToComplex2D | LinearOperator | _spbase

_Which: TypeAlias = Literal["LM", "SM", "LR", "SR", "LI", "SI"]
_OPpart: TypeAlias = Literal["r", "i"]
_Mode: TypeAlias = Literal["normal", "buckling", "cayley"]

###

class ArpackError(RuntimeError):
    def __init__(self, /, info: _KT, infodict: Mapping[_KT, str] = ...) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Final[onp.Array1D[np.float64 | np.complex128]]
    eigenvectors: Final[onp.Array2D[np.float64]]
    def __init__(
        self,
        /,
        msg: str,
        eigenvalues: onp.Array1D[np.float64 | np.complex128],
        eigenvectors: onp.Array2D[np.float64],
    ) -> None: ...

#
@overload  # returns_eigenvectors: truthy (default)
def eigs(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToRealMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: Truthy = True,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> tuple[onp.Array1D[np.complex128], onp.Array2D[np.float64]]: ...
@overload  # returns_eigenvectors: falsy (positional)
def eigs(
    A: _ToComplexMatrix,
    k: int,
    M: _ToRealMatrix | None,
    sigma: onp.ToComplex | None,
    which: _Which,
    v0: onp.ToFloat1D | None,
    ncv: int | None,
    maxiter: int | None,
    tol: float,
    return_eigenvectors: Falsy,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload  # returns_eigenvectors: falsy (keyword)
def eigs(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToRealMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: Falsy,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    OPpart: _OPpart | None = None,
) -> onp.Array1D[np.complex128]: ...

#
@overload  # returns_eigenvectors: truthy (default)
def eigsh(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToRealMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: Truthy = True,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    mode: _Mode = "normal",
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload  # returns_eigenvectors: falsy (positional)
def eigsh(
    A: _ToComplexMatrix,
    k: int,
    M: _ToRealMatrix | None,
    sigma: onp.ToComplex | None,
    which: _Which,
    v0: onp.ToFloat1D | None,
    ncv: int | None,
    maxiter: int | None,
    tol: float,
    return_eigenvectors: Falsy,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    mode: _Mode = "normal",
) -> onp.Array1D[np.float64]: ...
@overload  # returns_eigenvectors: falsy (keyword)
def eigsh(
    A: _ToComplexMatrix,
    k: int = 6,
    M: _ToRealMatrix | None = None,
    sigma: onp.ToComplex | None = None,
    which: _Which = "LM",
    v0: onp.ToFloat1D | None = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    *,
    return_eigenvectors: Falsy,
    Minv: _ToRealMatrix | None = None,
    OPinv: _ToRealMatrix | None = None,
    mode: _Mode = "normal",
) -> onp.Array1D[np.float64]: ...
