from scipy._typing import Untyped
from scipy.linalg import LinAlgError as LinAlgError
from scipy.sparse import (
    SparseEfficiencyWarning as SparseEfficiencyWarning,
    csc_matrix as csc_matrix,
    diags as diags,
    eye as eye,
    issparse as issparse,
)
from scipy.sparse._sputils import (
    convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy,
    is_pydata_spmatrix as is_pydata_spmatrix,
)

noScikit: bool
useUmfpack: Untyped

class MatrixRankWarning(UserWarning): ...

def use_solver(**kwargs): ...
def spsolve(A, b, permc_spec: Untyped | None = None, use_umfpack: bool = True) -> Untyped: ...
def splu(
    A,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def spilu(
    A,
    drop_tol: Untyped | None = None,
    fill_factor: Untyped | None = None,
    drop_rule: Untyped | None = None,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def factorized(A) -> Untyped: ...
def spsolve_triangular(
    A, b, lower: bool = True, overwrite_A: bool = False, overwrite_b: bool = False, unit_diagonal: bool = False
) -> Untyped: ...
