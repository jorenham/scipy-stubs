from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._lib._util import ComplexWarning as ComplexWarning
from scipy._typing import Untyped

def ldl(A, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True) -> Untyped: ...
