from scipy._typing import Untyped
from ._group_columns import group_dense as group_dense, group_sparse as group_sparse

def group_columns(A, order: int = 0) -> Untyped: ...
def approx_derivative(
    fun,
    x0,
    method: str = "3-point",
    rel_step: Untyped | None = None,
    abs_step: Untyped | None = None,
    f0: Untyped | None = None,
    bounds=...,
    sparsity: Untyped | None = None,
    as_linear_operator: bool = False,
    args=(),
    kwargs: Untyped | None = None,
) -> Untyped: ...
def check_derivative(fun, jac, x0, bounds=..., args=(), kwargs: Untyped | None = None) -> Untyped: ...
