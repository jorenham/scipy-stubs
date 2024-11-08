from collections.abc import Callable, Iterable, Mapping
from typing import Any, Concatenate, Final, Literal, Protocol, TypeAlias, type_check_only

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyInt, AnyReal
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

_Array_1d_f8: TypeAlias = onpt.Array[tuple[int], np.float64]
_Array_nd_f8: TypeAlias = onpt.Array[onpt.AtLeast1D, np.float64]

_LeastSquaresMethod: TypeAlias = Literal["trf", "dogbox", "lm"]
_JacMethod: TypeAlias = Literal["2-point", "3-point", "cs"]
_XScaleMethod: TypeAlias = Literal["jac"]
_LossMethod: TypeAlias = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]

_ResidFunction: TypeAlias = Callable[Concatenate[_Array_1d_f8, ...], _ArrayLikeFloat_co]
_JacFunction: TypeAlias = Callable[Concatenate[_Array_1d_f8, ...], _ArrayLikeFloat_co | _spbase | LinearOperator]

@type_check_only
class _LossFunction(Protocol):
    def __call__(self, x: _Array_1d_f8, /, *, cost_only: op.CanBool | None = None) -> _ArrayLikeFloat_co: ...

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
    x0: _ArrayLikeFloat_co,
    jac: _JacMethod | _JacFunction = "2-point",
    bounds: tuple[_ArrayLikeFloat_co, _ArrayLikeFloat_co] | Bounds = ...,
    method: _LeastSquaresMethod = "trf",
    ftol: AnyReal | None = 1e-08,
    xtol: AnyReal | None = 1e-08,
    gtol: AnyReal | None = 1e-08,
    x_scale: _ArrayLikeFloat_co | _XScaleMethod = 1.0,
    loss: _LossMethod | _LossFunction = "linear",
    f_scale: AnyReal = 1.0,
    diff_step: _ArrayLikeFloat_co | None = None,
    tr_solver: Literal["exact", "lsmr"] | None = None,
    tr_options: Mapping[str, object] = {},
    jac_sparsity: _ArrayLikeFloat_co | _spbase | None = None,
    max_nfev: AnyInt | None = None,
    verbose: Literal[0, 1, 2] = 0,
    args: Iterable[object] = (),
    kwargs: Mapping[str, object] = {},
) -> _OptimizeResult: ...

# undocumented
def call_minpack(
    fun: _ResidFunction,
    x0: _ArrayLikeFloat_co,
    jac: _JacMethod | _JacFunction | None,
    ftol: AnyReal,
    xtol: AnyReal,
    gtol: AnyReal,
    max_nfev: AnyInt | None,
    x_scale: AnyReal | _Array_nd_f8,
    diff_step: _ArrayLikeFloat_co | None,
) -> _OptimizeResult: ...

# undocumented
def prepare_bounds(bounds: Iterable[_ArrayLikeFloat_co], n: op.CanIndex) -> tuple[_Array_nd_f8, _Array_nd_f8]: ...

# undocumented
def check_tolerance(
    ftol: AnyReal | None,
    xtol: AnyReal | None,
    gtol: AnyReal | None,
    method: _LeastSquaresMethod,
) -> tuple[AnyReal, AnyReal, AnyReal]: ...
def check_x_scale(x_scale: _ArrayLikeFloat_co | _XScaleMethod, x0: npt.NDArray[np.floating[Any]]) -> _Array_nd_f8: ...
def check_jac_sparsity(jac_sparsity: _ArrayLikeFloat_co | _spbase | None, m: AnyInt, n: AnyInt) -> _Array_1d_f8: ...

# undocumented
def huber(z: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], cost_only: op.CanBool) -> None: ...
def soft_l1(z: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], cost_only: op.CanBool) -> None: ...
def cauchy(z: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], cost_only: op.CanBool) -> None: ...
def arctan(z: AnyReal, rho: npt.NDArray[np.float64], cost_only: op.CanBool) -> None: ...

# undocumented
def construct_loss_function(
    m: op.CanIndex,
    loss: _LossMethod | _LossFunction,
    f_scale: AnyReal,
) -> Callable[[_Array_1d_f8], _Array_1d_f8]: ...
