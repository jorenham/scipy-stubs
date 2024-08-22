from ._misc import LinAlgError as LinAlgError, LinAlgWarning as LinAlgWarning
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

def qz(
    A,
    B,
    output: str = "real",
    lwork: Untyped | None = None,
    sort: Untyped | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> Untyped: ...
def ordqz(
    A, B, sort: str = "lhp", output: str = "real", overwrite_a: bool = False, overwrite_b: bool = False, check_finite: bool = True
) -> Untyped: ...
