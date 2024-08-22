from ._basic import matrix_balance as matrix_balance, solve as solve, solve_triangular as solve_triangular
from ._decomp_lu import lu as lu
from ._decomp_qr import qr as qr
from ._decomp_qz import ordqz as ordqz
from ._decomp_schur import schur as schur
from ._special_matrices import block_diag as block_diag, kron as kron
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

def solve_sylvester(a, b, q) -> Untyped: ...
def solve_continuous_lyapunov(a, q) -> Untyped: ...

solve_lyapunov = solve_continuous_lyapunov

def solve_discrete_lyapunov(a, q, method: Untyped | None = None) -> Untyped: ...
def solve_continuous_are(a, b, q, r, e: Untyped | None = None, s: Untyped | None = None, balanced: bool = True) -> Untyped: ...
def solve_discrete_are(a, b, q, r, e: Untyped | None = None, s: Untyped | None = None, balanced: bool = True) -> Untyped: ...
