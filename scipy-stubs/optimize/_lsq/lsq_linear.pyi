from .bvls import bvls as bvls
from .common import compute_grad as compute_grad, in_bounds as in_bounds
from .trf_linear import trf_linear as trf_linear
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as OptimizeResult
from scipy.optimize._minimize import Bounds as Bounds
from scipy.sparse import csr_matrix as csr_matrix, issparse as issparse
from scipy.sparse.linalg import LinearOperator as LinearOperator, lsmr as lsmr

def prepare_bounds(bounds, n) -> Untyped: ...

TERMINATION_MESSAGES: Untyped

def lsq_linear(
    A,
    b,
    bounds=...,
    method: str = "trf",
    tol: float = 1e-10,
    lsq_solver: Untyped | None = None,
    lsmr_tol: Untyped | None = None,
    max_iter: Untyped | None = None,
    verbose: int = 0,
    *,
    lsmr_maxiter: Untyped | None = None,
) -> Untyped: ...
