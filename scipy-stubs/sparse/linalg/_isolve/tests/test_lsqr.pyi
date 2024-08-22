from scipy._typing import Untyped
from scipy.sparse.linalg import lsqr as lsqr

n: int
G: Untyped
normal: Untyped
norm: Untyped
gg: Untyped
hh: Untyped
b: Untyped
tol: float
atol_test: float
rtol_test: float
show: bool
maxit: Untyped

def test_lsqr_basic(): ...
def test_gh_2466(): ...
def test_well_conditioned_problems(): ...
def test_b_shapes(): ...
def test_initialization(): ...
