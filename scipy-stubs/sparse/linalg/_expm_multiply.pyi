from scipy._typing import Untyped
from scipy.linalg._decomp_qr import qr as qr
from scipy.sparse._sputils import is_pydata_spmatrix as is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator as aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator as IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest as onenormest

def traceest(A, m3, seed: Untyped | None = None) -> Untyped: ...
def expm_multiply(
    A,
    B,
    start: Untyped | None = None,
    stop: Untyped | None = None,
    num: Untyped | None = None,
    endpoint: Untyped | None = None,
    traceA: Untyped | None = None,
) -> Untyped: ...

class LazyOperatorNormInfo:
    def __init__(self, A, A_1_norm: Untyped | None = None, ell: int = 2, scale: int = 1): ...
    def set_scale(self, scale): ...
    def onenorm(self) -> Untyped: ...
    def d(self, p) -> Untyped: ...
    def alpha(self, p) -> Untyped: ...
