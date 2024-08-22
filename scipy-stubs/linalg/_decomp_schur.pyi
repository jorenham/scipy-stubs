from ._decomp import eigvals as eigvals
from ._misc import LinAlgError as LinAlgError
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

def schur(
    a,
    output: str = "real",
    lwork: Untyped | None = None,
    overwrite_a: bool = False,
    sort: Untyped | None = None,
    check_finite: bool = True,
) -> Untyped: ...

eps: Untyped
feps: Untyped

def rsf2csf(T, Z, check_finite: bool = True) -> Untyped: ...
