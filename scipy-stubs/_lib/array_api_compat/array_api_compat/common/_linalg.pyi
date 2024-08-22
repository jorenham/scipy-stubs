from typing import Literal, NamedTuple

from .._internal import get_xp as get_xp
from ._aliases import (
    isdtype as isdtype,
    matmul as matmul,
    matrix_transpose as matrix_transpose,
    tensordot as tensordot,
    vecdot as vecdot,
)
from ._typing import ndarray as ndarray
from scipy._typing import Untyped

def cross(x1: ndarray, x2: ndarray, /, xp, *, axis: int = -1, **kwargs) -> ndarray: ...
def outer(x1: ndarray, x2: ndarray, /, xp, **kwargs) -> ndarray: ...

class EighResult(NamedTuple):
    eigenvalues: ndarray
    eigenvectors: ndarray

class QRResult(NamedTuple):
    Q: ndarray
    R: ndarray

class SlogdetResult(NamedTuple):
    sign: ndarray
    logabsdet: ndarray

class SVDResult(NamedTuple):
    U: ndarray
    S: ndarray
    Vh: ndarray

def eigh(x: ndarray, /, xp, **kwargs) -> EighResult: ...
def qr(x: ndarray, /, xp, *, mode: Literal["reduced", "complete"] = "reduced", **kwargs) -> QRResult: ...
def slogdet(x: ndarray, /, xp, **kwargs) -> SlogdetResult: ...
def svd(x: ndarray, /, xp, *, full_matrices: bool = True, **kwargs) -> SVDResult: ...
def cholesky(x: ndarray, /, xp, *, upper: bool = False, **kwargs) -> ndarray: ...
def matrix_rank(x: ndarray, /, xp, *, rtol: float | ndarray | None = None, **kwargs) -> ndarray: ...
def pinv(x: ndarray, /, xp, *, rtol: float | ndarray | None = None, **kwargs) -> ndarray: ...
def matrix_norm(
    x: ndarray, /, xp, *, keepdims: bool = False, ord: int | float | Literal["fro", "nuc"] | None = "fro"
) -> ndarray: ...
def svdvals(x: ndarray, /, xp) -> ndarray | tuple[ndarray, ...]: ...
def vector_norm(
    x: ndarray, /, xp, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False, ord: int | float | None = 2
) -> ndarray: ...
def diagonal(x: ndarray, /, xp, *, offset: int = 0, **kwargs) -> ndarray: ...
def trace(x: ndarray, /, xp, *, offset: int = 0, dtype: Untyped | None = None, **kwargs) -> ndarray: ...
