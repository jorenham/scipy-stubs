from scipy._typing import Untyped
from scipy.linalg._flapack import *

HAS_ILP64: bool
empty_module: Untyped
p1: Untyped
p2: Untyped

def backtickrepl(m) -> Untyped: ...
def get_lapack_funcs(names, arrays=(), dtype: Untyped | None = None, ilp64: bool = False) -> Untyped: ...
