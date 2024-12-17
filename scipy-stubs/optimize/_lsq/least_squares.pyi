from collections.abc import Callable, Iterable, Mapping
from typing import Any, Concatenate, Final, Literal, Protocol, TypeAlias, type_check_only

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float1ND: TypeAlias = onp.Array[onp.AtLeast1D, np.float64]
_ToJac2D: TypeAlias = onp.ToFloat2D | _spbase

_LeastSquaresMethod: TypeAlias = Literal["trf", "dogbox", "lm"]
_JacMethod: TypeAlias = Literal["2-point", "3-point", "cs"]
_Jac: TypeAlias = _JacFunction | _JacMethod
_XScaleMethod: TypeAlias = Literal["jac"]
_XScale: TypeAlias = onp.ToFloat | onp.ToFloatND | _XScaleMethod
_LossMethod: TypeAlias = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
_Loss: TypeAlias = _LossFunction | _LossMethod

_ResidFunction: TypeAlias = Callable[Concatenate[_Float1D, ...], onp.ToFloat1D]
_JacFunction: TypeAlias = Callable[Concatenate[_Float1D, ...], _ToJac2D | LinearOperator]

@type_check_only
class _LossFunction(Protocol):
    def __call__(self, x: _Float1D, /, *, cost_only: op.CanBool | None = None) -> onp.ToFloat1D: ...

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: object
    info: Mapping[str, object]
    status: object

###

TERMINATION_MESSAGES: Final[dict[Literal[-1, 0, 1, 2, 3, 4], str]] = ...
FROM_MINPACK_TO_COMMON: Final[dict[Literal[0, 1, 2, 3, 4, 5], Literal[-1, 2, 3, 4, 1, 0]]] = ...
IMPLEMENTED_LOSSES: Final[dict[str, _ResidFunction]] = ...

#
def least_squares(
    fun: _ResidFunction,
    x0: onp.ToFloat1D,
    jac: _Jac = "2-point",
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _LeastSquaresMethod = "trf",
    ftol: onp.ToFloat | None = 1e-08,
    xtol: onp.ToFloat | None = 1e-08,
    gtol: onp.ToFloat | None = 1e-08,
    x_scale: _XScale = 1.0,
    loss: _Loss = "linear",
    f_scale: onp.ToFloat = 1.0,
    diff_step: onp.ToFloat1D | None = None,
    tr_solver: Literal["exact", "lsmr"] | None = None,
    tr_options: Mapping[str, object] | None = None,
    jac_sparsity: _ToJac2D | None = None,
    max_nfev: onp.ToInt | None = None,
    verbose: Literal[0, 1, 2] = 0,
    args: Iterable[object] = (),
    kwargs: Mapping[str, object] | None = None,
) -> _OptimizeResult: ...

# undocumented
def call_minpack(
    fun: _ResidFunction,
    x0: onp.ToFloat1D,
    jac: _Jac | None,
    ftol: onp.ToFloat,
    xtol: onp.ToFloat,
    gtol: onp.ToFloat,
    max_nfev: onp.ToInt | None,
    x_scale: onp.ToFloat | _Float1ND,
    diff_step: onp.ToFloat1D | None,
) -> _OptimizeResult: ...

# undocumented
def prepare_bounds(bounds: Iterable[onp.ToFloat | onp.ToFloat1D], n: op.CanIndex) -> tuple[_Float1D, _Float1D]: ...

# undocumented
def check_tolerance(
    ftol: onp.ToFloat | None,
    xtol: onp.ToFloat | None,
    gtol: onp.ToFloat | None,
    method: _LeastSquaresMethod,
) -> tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat]: ...
def check_x_scale(x_scale: _XScale, x0: onp.ArrayND[np.floating[Any]]) -> _Float1ND: ...
def check_jac_sparsity(jac_sparsity: _ToJac2D | None, m: onp.ToInt, n: onp.ToInt) -> _Float1D: ...

# undocumented
def huber(z: onp.ArrayND[np.float64], rho: onp.ArrayND[np.float64], cost_only: op.CanBool) -> None: ...
def soft_l1(z: onp.ArrayND[np.float64], rho: onp.ArrayND[np.float64], cost_only: op.CanBool) -> None: ...
def cauchy(z: onp.ArrayND[np.float64], rho: onp.ArrayND[np.float64], cost_only: op.CanBool) -> None: ...
def arctan(z: onp.ToFloat, rho: onp.ArrayND[np.float64], cost_only: op.CanBool) -> None: ...

# undocumented
def construct_loss_function(m: op.CanIndex, loss: _Loss, f_scale: onp.ToFloat) -> Callable[[_Float1D], _Float1D]: ...
