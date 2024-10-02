from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, Final, Protocol, TypeAlias, type_check_only

import numpy as np
import optype.numpy as onpt
from scipy._typing import Untyped
from ._constraints import Bounds
from ._hessian_update_strategy import HessianUpdateStrategy
from ._optimize import OptimizeResult
from ._typing import Constraint, MethodJac, MethodMimimize, MethodMinimizeScalar

__all__ = ["minimize", "minimize_scalar"]

_Array_f_1d: TypeAlias = onpt.Array[tuple[int], np.floating[Any]]
_ArrayLike_f_1d: TypeAlias = Sequence[float | np.floating[Any]] | onpt.CanArray[tuple[int], np.dtype[np.floating[Any]]]
_ArrayLike_f_2d: TypeAlias = Sequence[_ArrayLike_f_1d] | onpt.CanArray[tuple[int], np.dtype[np.floating[Any]]]

_Bound: TypeAlias = tuple[float | None, float | None]

@type_check_only
class _MinimizeCallback(Protocol):
    def __call__(self, /, intermediate_result: OptimizeResult) -> None: ...

@type_check_only
class _MinimizeCallbackXk(Protocol):
    def __call__(self, /, xk: _Array_f_1d) -> None: ...

###

MINIMIZE_METHODS: Final[Sequence[MethodMimimize]] = ...
MINIMIZE_METHODS_NEW_CB: Final[Sequence[MethodMimimize]] = ...
MINIMIZE_SCALAR_METHODS: Final[Sequence[MethodMinimizeScalar]] = ...

def minimize(
    fun: Callable[Concatenate[_Array_f_1d, ...], float | np.floating[Any]],
    x0: _ArrayLike_f_1d,
    args: tuple[object, ...] = (),
    method: MethodMimimize | Callable[..., OptimizeResult] | None = None,
    jac: MethodJac | Callable[Concatenate[float, ...], _ArrayLike_f_1d] | None = None,
    # TODO: also allow the callable to return a `LinearOperator` or a sparse matrix
    hess: MethodJac | HessianUpdateStrategy | Callable[Concatenate[float, ...], _ArrayLike_f_2d] | None = None,
    hessp: Callable[Concatenate[_Array_f_1d, _Array_f_1d, ...], _ArrayLike_f_1d] | None = None,
    bounds: Sequence[_Bound] | Bounds | None = None,
    constraints: Constraint | Sequence[Constraint] = (),
    tol: float | None = None,
    callback: _MinimizeCallback | _MinimizeCallbackXk | None = None,
    # TODO: TypedDict (dependent on `method`)
    options: Mapping[str, object] | None = None,
) -> OptimizeResult: ...
def minimize_scalar(
    fun: Callable[Concatenate[float, ...], float | np.floating[Any]],
    bracket: Sequence[tuple[float, float] | tuple[float, float, float]] | None = None,
    bounds: Sequence[_Bound] | None = None,
    args: tuple[object, ...] = (),
    method: MethodMinimizeScalar | Callable[..., OptimizeResult] | None = None,
    tol: float | None = None,
    options: Mapping[str, object] | None = None,
) -> Untyped: ...
def standardize_bounds(  # undocumented
    bounds: Sequence[_Bound] | Bounds,
    x0: _ArrayLike_f_1d,
    meth: MethodMimimize,
) -> Bounds | list[_Bound]: ...
def standardize_constraints(  # undocumented
    constraints: Constraint | Sequence[Constraint],
    x0: _ArrayLike_f_1d,
    meth: object,  # unused
) -> list[Constraint]: ...
