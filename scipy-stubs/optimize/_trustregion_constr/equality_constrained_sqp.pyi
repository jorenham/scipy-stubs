from typing import Any, TypeVar

import numpy as np
import optype.numpy as onpt
from scipy._typing import Untyped, UntypedCallable
from scipy.sparse import dia_matrix

__all__ = ["equality_constrained_sqp"]

_StateT = TypeVar("_StateT")

def default_scaling(x: onpt.Array[tuple[int]]) -> dia_matrix: ...
def equality_constrained_sqp(
    fun_and_constr: Untyped,
    grad_and_jac: Untyped,
    lagr_hess: Untyped,
    x0: onpt.Array[tuple[int], np.floating[Any]],
    fun0: Untyped,
    grad0: Untyped,
    constr0: Untyped,
    jac0: Untyped,
    stop_criteria: Untyped,
    state: _StateT,
    initial_penalty: Untyped,
    initial_trust_radius: Untyped,
    factorization_method: Untyped,
    trust_lb: Untyped | None = None,
    trust_ub: Untyped | None = None,
    scaling: UntypedCallable = ...,
) -> tuple[onpt.Array[tuple[int], np.floating[Any]], _StateT]: ...
