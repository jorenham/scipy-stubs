from collections.abc import Callable, Iterable
from typing import Concatenate, Final, Literal, TypeAlias, TypedDict, type_check_only
from typing_extensions import NotRequired, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.optimize._differentiable_functions import LinearVectorFunction, VectorFunction
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_T = TypeVar("_T")
_ShapeT_co = TypeVar("_ShapeT_co", bound=onp.AtLeast1D, default=onp.AtLeast1D, covariant=True)
_SCT = TypeVar("_SCT", bound=np.generic, default=np.float64)

_Tuple2: TypeAlias = tuple[_T, _T]
_Sparse: TypeAlias = sparray | spmatrix
_Vector: TypeAlias = onp.Array1D[_SCT]
_Matrix: TypeAlias = onp.Array2D[_SCT] | _Sparse

_MethodJac: TypeAlias = Literal["2-point", "3-point", "cs"]

@type_check_only
class _OldConstraint(TypedDict):
    type: Literal["eq", "ineq"]
    fun: Callable[Concatenate[_Vector, ...], onp.ToFloat1D]
    jac: NotRequired[Callable[Concatenate[_Vector, ...], onp.ToFloat2D | _Sparse]]
    args: NotRequired[tuple[object, ...]]

@type_check_only
class _BaseConstraint:
    keep_feasible: Final[_Vector[np.bool_]]

@type_check_only
class _Constraint(_BaseConstraint):
    lb: Final[_Vector]
    ub: Final[_Vector]

###

class Bounds(_Constraint):
    def __init__(
        self,
        /,
        lb: onp.ToFloat | onp.ToFloat1D = ...,
        ub: onp.ToFloat | onp.ToFloat1D = ...,
        keep_feasible: onp.ToBool | onp.ToBool1D = False,
    ) -> None: ...
    def residual(self, /, x: onp.ToFloat1D) -> _Tuple2[onp.Array[_ShapeT_co, np.float64]]: ...

class LinearConstraint(_Constraint):
    A: Final[_Matrix]

    def __init__(
        self,
        /,
        A: onp.ToFloat2D | _Sparse,
        lb: onp.ToFloat | onp.ToFloat1D = ...,
        ub: onp.ToFloat | onp.ToFloat1D = ...,
        keep_feasible: onp.ToBool | onp.ToBool1D = False,
    ) -> None: ...
    def residual(self, /, x: onp.ToFloat1D) -> _Tuple2[_Vector]: ...

class NonlinearConstraint(_Constraint):
    fun: Final[Callable[[_Vector], onp.ToFloat1D]]
    finite_diff_rel_step: Final[onp.ToFloat | onp.ToFloat1D | None]
    finite_diff_jac_sparsity: Final[onp.ToFloat2D | _Sparse | None]
    jac: Final[Callable[[_Vector], onp.ToFloat2D | _Sparse] | _MethodJac]
    hess: Final[Callable[[_Vector], onp.ToFloat2D | _Sparse | LinearOperator] | _MethodJac | HessianUpdateStrategy | None]

    def __init__(
        self,
        /,
        fun: Callable[[_Vector], onp.ToFloat1D],
        lb: onp.ToFloat | onp.ToFloat1D,
        ub: onp.ToFloat | onp.ToFloat1D,
        jac: Callable[[_Vector], onp.ToFloat2D | _Sparse] | _MethodJac = "2-point",
        hess: Callable[[_Vector], onp.ToFloat2D | _Sparse | LinearOperator] | _MethodJac | HessianUpdateStrategy | None = ...,
        keep_feasible: onp.ToBool | onp.ToBool1D = False,
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None = None,
        finite_diff_jac_sparsity: onp.ToFloat2D | _Sparse | None = None,
    ) -> None: ...

class PreparedConstraint(_BaseConstraint):
    fun: Final[VectorFunction | LinearVectorFunction]
    bounds: Final[_Tuple2[_Vector]]

    def __init__(
        self,
        /,
        constraint: _Constraint,
        x0: onp.ToFloat1D,
        sparse_jacobian: bool | None = None,
        finite_diff_bounds: _Tuple2[onp.ToFloat | onp.ToFloat2D] = ...,
    ) -> None: ...
    def violation(self, /, x: onp.ToFloat1D) -> _Vector: ...

def new_bounds_to_old(lb: onp.ToFloat1D, ub: onp.ToFloat1D, n: op.CanIndex) -> list[_Tuple2[float]]: ...
def old_bound_to_new(bounds: Iterable[_Tuple2[float]]) -> _Tuple2[_Vector]: ...
def strict_bounds(lb: onp.ToFloat1D, ub: onp.ToFloat1D, keep_feasible: onp.ToBool1D, n_vars: op.CanIndex) -> _Tuple2[_Vector]: ...
def new_constraint_to_old(con: _BaseConstraint, x0: onp.ToFloatND) -> list[_OldConstraint]: ...
def old_constraint_to_new(ic: int, con: _OldConstraint) -> NonlinearConstraint: ...
