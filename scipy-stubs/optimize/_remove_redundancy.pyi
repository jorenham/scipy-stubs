from scipy._typing import Untyped
from scipy.linalg import svd as svd
from scipy.linalg.blas import dtrsm as dtrsm
from scipy.linalg.interpolative import interp_decomp as interp_decomp

def bg_update_dense(plu, perm_r, v, j) -> Untyped: ...
