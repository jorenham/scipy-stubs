from ._misc import LinAlgError as LinAlgError, norm as norm
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

def eig(
    a,
    b: Untyped | None = None,
    left: bool = False,
    right: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> Untyped: ...
def eigh(
    a,
    b: Untyped | None = None,
    *,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: int = 1,
    check_finite: bool = True,
    subset_by_index: Untyped | None = None,
    subset_by_value: Untyped | None = None,
    driver: Untyped | None = None,
) -> Untyped: ...
def eig_banded(
    a_band,
    lower: bool = False,
    eigvals_only: bool = False,
    overwrite_a_band: bool = False,
    select: str = "a",
    select_range: Untyped | None = None,
    max_ev: int = 0,
    check_finite: bool = True,
) -> Untyped: ...
def eigvals(
    a, b: Untyped | None = None, overwrite_a: bool = False, check_finite: bool = True, homogeneous_eigvals: bool = False
) -> Untyped: ...
def eigvalsh(
    a,
    b: Untyped | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: int = 1,
    check_finite: bool = True,
    subset_by_index: Untyped | None = None,
    subset_by_value: Untyped | None = None,
    driver: Untyped | None = None,
) -> Untyped: ...
def eigvals_banded(
    a_band,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: str = "a",
    select_range: Untyped | None = None,
    check_finite: bool = True,
) -> Untyped: ...
def eigvalsh_tridiagonal(
    d,
    e,
    select: str = "a",
    select_range: Untyped | None = None,
    check_finite: bool = True,
    tol: float = 0.0,
    lapack_driver: str = "auto",
) -> Untyped: ...
def eigh_tridiagonal(
    d,
    e,
    eigvals_only: bool = False,
    select: str = "a",
    select_range: Untyped | None = None,
    check_finite: bool = True,
    tol: float = 0.0,
    lapack_driver: str = "auto",
) -> Untyped: ...
def hessenberg(a, calc_q: bool = False, overwrite_a: bool = False, check_finite: bool = True) -> Untyped: ...
def cdf2rdf(w, v) -> Untyped: ...
