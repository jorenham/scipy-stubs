from scipy._typing import Untyped
from scipy.sparse import issparse as issparse
from scipy.sparse._sputils import (
    convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy,
    is_pydata_spmatrix as is_pydata_spmatrix,
)
from scipy.sparse.linalg import LinearOperator as LinearOperator

def laplacian(
    csgraph,
    normed: bool = False,
    return_diag: bool = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: str = "array",
    dtype: Untyped | None = None,
    symmetrized: bool = False,
) -> Untyped: ...
