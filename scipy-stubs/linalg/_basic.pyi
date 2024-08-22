from ._cythonized_array_utils import (
    bandwidth as bandwidth,
    find_det_from_lu as find_det_from_lu,
    ishermitian as ishermitian,
    issymmetric as issymmetric,
)
from ._misc import LinAlgError as LinAlgError, LinAlgWarning as LinAlgWarning
from ._solve_toeplitz import levinson as levinson
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

lapack_cast_dict: Untyped

def solve(
    a,
    b,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: Untyped | None = None,
    transposed: bool = False,
) -> Untyped: ...
def solve_triangular(
    a, b, trans: int = 0, lower: bool = False, unit_diagonal: bool = False, overwrite_b: bool = False, check_finite: bool = True
) -> Untyped: ...
def solve_banded(l_and_u, ab, b, overwrite_ab: bool = False, overwrite_b: bool = False, check_finite: bool = True) -> Untyped: ...
def solveh_banded(
    ab, b, overwrite_ab: bool = False, overwrite_b: bool = False, lower: bool = False, check_finite: bool = True
) -> Untyped: ...
def solve_toeplitz(c_or_cr, b, check_finite: bool = True) -> Untyped: ...
def solve_circulant(
    c, b, singular: str = "raise", tol: Untyped | None = None, caxis: int = -1, baxis: int = 0, outaxis: int = 0
) -> Untyped: ...
def inv(a, overwrite_a: bool = False, check_finite: bool = True) -> Untyped: ...
def det(a, overwrite_a: bool = False, check_finite: bool = True) -> Untyped: ...
def lstsq(
    a,
    b,
    cond: Untyped | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: Untyped | None = None,
) -> Untyped: ...
def pinv(
    a, *, atol: Untyped | None = None, rtol: Untyped | None = None, return_rank: bool = False, check_finite: bool = True
) -> Untyped: ...
def pinvh(
    a,
    atol: Untyped | None = None,
    rtol: Untyped | None = None,
    lower: bool = True,
    return_rank: bool = False,
    check_finite: bool = True,
) -> Untyped: ...
def matrix_balance(A, permute: bool = True, scale: bool = True, separate: bool = False, overwrite_a: bool = False) -> Untyped: ...
def matmul_toeplitz(c_or_cr, x, check_finite: bool = False, workers: Untyped | None = None) -> Untyped: ...
