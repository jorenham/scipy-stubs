from typing import Literal

from dask.array import matmul as matmul, outer as outer, tensordot as tensordot, trace as trace
from dask.array.linalg import *

from ..._internal import get_xp as get_xp
from ...common._typing import Array as Array
from ._aliases import matrix_transpose as matrix_transpose, vecdot as vecdot
from scipy._typing import Untyped

linalg_all: Untyped
EighResult: Untyped
QRResult: Untyped
SlogdetResult: Untyped
SVDResult: Untyped

def qr(x: Array, mode: Literal["reduced", "complete"] = "reduced", **kwargs) -> QRResult: ...

cholesky: Untyped
matrix_rank: Untyped
matrix_norm: Untyped

def svd(x: Array, full_matrices: bool = True, **kwargs) -> SVDResult: ...
def svdvals(x: Array) -> Array: ...

vector_norm: Untyped
diagonal: Untyped
