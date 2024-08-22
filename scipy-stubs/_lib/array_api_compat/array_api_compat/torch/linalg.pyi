from typing import Literal

from torch import dtype as Dtype, outer as outer
from torch.linalg import *

from ._aliases import matmul as matmul, matrix_transpose as matrix_transpose, sum as sum, tensordot as tensordot
from scipy._typing import Untyped

array: Untyped
inf: Untyped
linalg_all: Untyped

def cross(x1: array, x2: array, /, *, axis: int = -1) -> array: ...
def vecdot(x1: array, x2: array, /, *, axis: int = -1, **kwargs) -> array: ...
def solve(x1: array, x2: array, /, **kwargs) -> array: ...
def trace(x: array, /, *, offset: int = 0, dtype: Dtype | None = None) -> array: ...
def vector_norm(
    x: array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float | Literal[inf, None] = 2,
    **kwargs,
) -> array: ...
