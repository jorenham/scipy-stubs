from ..sparse import (
    coo_matrix as coo_matrix,
    csc_matrix as csc_matrix,
    csr_matrix as csr_matrix,
    find as find,
    issparse as issparse,
)
from ._group_columns import group_dense as group_dense, group_sparse as group_sparse
from scipy._lib._array_api import array_namespace as array_namespace, xp_atleast_nd as xp_atleast_nd
from scipy._typing import Untyped
from scipy.sparse.linalg import LinearOperator as LinearOperator

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
