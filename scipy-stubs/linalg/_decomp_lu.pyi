from ._decomp_lu_cython import lu_dispatcher as lu_dispatcher
from ._misc import LinAlgWarning as LinAlgWarning
from .lapack import get_lapack_funcs as get_lapack_funcs
from scipy._typing import Untyped

lapack_cast_dict: Untyped

def lu_factor(a, overwrite_a: bool = False, check_finite: bool = True) -> Untyped: ...
def lu_solve(lu_and_piv, b, trans: int = 0, overwrite_b: bool = False, check_finite: bool = True) -> Untyped: ...
def lu(a, permute_l: bool = False, overwrite_a: bool = False, check_finite: bool = True, p_indices: bool = False) -> Untyped: ...
