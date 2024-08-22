from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

def safecall(f, name, *args, **kwargs) -> Untyped: ...
def qr(
    a,
    overwrite_a: bool = False,
    lwork: Untyped | None = None,
    mode: str = "full",
    pivoting: bool = False,
    check_finite: bool = True,
) -> Untyped: ...
def qr_multiply(
    a,
    c,
    mode: str = "right",
    pivoting: bool = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> Untyped: ...
def rq(a, overwrite_a: bool = False, lwork: Untyped | None = None, mode: str = "full", check_finite: bool = True) -> Untyped: ...
