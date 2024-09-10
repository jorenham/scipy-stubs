from collections.abc import Callable
from typing import TypeAlias, TypeVar
from typing_extensions import Self

import numpy as np
import optype as op
import optype.numpy as onpt
from scipy._typing import Untyped
from scipy.optimize._constraints import PreparedConstraint
from scipy.sparse import csr_matrix
from scipy.sparse.linalg._interface import LinearOperator

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]

_FunConstr: TypeAlias = Callable[[onpt.Array[tuple[int], np.float64]], _Tuple2[onpt.Array[tuple[int], np.float64]]]
_FunJac: TypeAlias = Callable[[onpt.Array[tuple[int], np.float64]], _Tuple2[onpt.Array[tuple[int, int], np.float64] | csr_matrix]]
_FunHess: TypeAlias = Callable[
    [onpt.Array[tuple[int], np.float64], onpt.Array[tuple[int], np.float64], onpt.Array[tuple[int], np.float64]],
    _Tuple2[onpt.Array[tuple[int, int], np.float64] | csr_matrix | LinearOperator],
]

# tighter than `Iterable[PreparedConstraint]` ;)
_PreparedConstraints: TypeAlias = op.CanIter[op.CanNext[CanonicalConstraint]]

class CanonicalConstraint:
    n_eq: int
    n_ineq: int
    fun: Untyped
    jac: Untyped
    hess: Untyped
    keep_feasible: Untyped
    def __init__(
        self,
        /,
        n_eq: int,
        n_ineq: int,
        fun: _FunConstr,
        jac: _FunJac,
        hess: _FunHess,
        keep_feasible: onpt.Array[tuple[int], np.bool_],
    ) -> None: ...
    @classmethod
    def from_PreparedConstraint(cls, constraint: PreparedConstraint) -> Self: ...
    @classmethod
    def empty(cls, n: op.CanIndex) -> Self: ...
    @classmethod
    def concatenate(cls, canonical_constraints: _PreparedConstraints, sparse_jacobian: bool | np.bool_) -> Self: ...

def initial_constraints_as_canonical(
    n: op.CanIndex,
    prepared_constraints: _PreparedConstraints,
    sparse_jacobian: bool | np.bool_,
) -> tuple[
    onpt.Array[onpt.AtMost2D, np.float64],
    onpt.Array[onpt.AtMost2D, np.float64],
    onpt.Array[tuple[int, int], np.float64] | csr_matrix,
    onpt.Array[tuple[int, int], np.float64] | csr_matrix,
]: ...
