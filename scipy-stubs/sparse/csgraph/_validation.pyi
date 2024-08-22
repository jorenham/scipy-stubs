from scipy._typing import Untyped
from scipy.sparse import csr_matrix as csr_matrix, issparse as issparse
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy
from scipy.sparse.csgraph._tools import (
    csgraph_from_dense as csgraph_from_dense,
    csgraph_from_masked as csgraph_from_masked,
    csgraph_masked_from_dense as csgraph_masked_from_dense,
    csgraph_to_dense as csgraph_to_dense,
)

DTYPE: Untyped

def validate_graph(
    csgraph,
    directed,
    dtype=...,
    csr_output: bool = True,
    dense_output: bool = True,
    copy_if_dense: bool = False,
    copy_if_sparse: bool = False,
    null_value_in: int = 0,
    null_value_out=...,
    infinity_null: bool = True,
    nan_null: bool = True,
) -> Untyped: ...
