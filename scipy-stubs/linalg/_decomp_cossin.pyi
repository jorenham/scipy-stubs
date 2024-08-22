from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped
from scipy.linalg import LinAlgError as LinAlgError, block_diag as block_diag

def cossin(
    X,
    p: Untyped | None = None,
    q: Untyped | None = None,
    separate: bool = False,
    swap_sign: bool = False,
    compute_u: bool = True,
    compute_vh: bool = True,
) -> Untyped: ...
