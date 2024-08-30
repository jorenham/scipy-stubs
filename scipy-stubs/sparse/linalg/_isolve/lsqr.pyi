from scipy._typing import Untyped
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy
from scipy.sparse.linalg._interface import aslinearoperator as aslinearoperator

eps: Untyped

def lsqr(
    A,
    b,
    damp: float = 0.0,
    atol: float = 1e-06,
    btol: float = 1e-06,
    conlim: float = ...,
    iter_lim: Untyped | None = None,
    show: bool = False,
    calc_var: bool = False,
    x0: Untyped | None = None,
) -> Untyped: ...
