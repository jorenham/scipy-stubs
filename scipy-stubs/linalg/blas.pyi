from scipy._typing import Untyped
from scipy.linalg._fblas import *

HAS_ILP64: bool
empty_module: Untyped

def find_best_blas_type(arrays=(), dtype: Untyped | None = None) -> Untyped: ...
def get_blas_funcs(names, arrays=(), dtype: Untyped | None = None, ilp64: bool = False) -> Untyped: ...
