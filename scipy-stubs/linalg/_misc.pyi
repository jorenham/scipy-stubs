from numpy.linalg import LinAlgError as LinAlgError

from .blas import get_blas_funcs as get_blas_funcs
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

class LinAlgWarning(RuntimeWarning): ...

def norm(
    a, ord: Untyped | None = None, axis: Untyped | None = None, keepdims: bool = False, check_finite: bool = True
) -> Untyped: ...
