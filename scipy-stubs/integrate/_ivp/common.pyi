from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.sparse import csc_matrix
from .base import DenseOutput

_SCT = TypeVar("_SCT", bound=np.generic)
_ToFloatT = TypeVar("_ToFloatT", bound=onp.ToFloat)

_Side: TypeAlias = Literal["left", "right"]
_Interpolants: TypeAlias = Sequence[DenseOutput]

_To1D: TypeAlias = Sequence[_SCT] | onp.CanArrayND[_SCT]
_ToFloat64: TypeAlias = np.float16 | np.float32 | np.float64 | np.integer[Any] | np.bool_

###

EPS: Final[float] = ...
NUM_JAC_DIFF_REJECT: Final[float] = ...
NUM_JAC_DIFF_SMALL: Final[float] = ...
NUM_JAC_DIFF_BIG: Final[float] = ...
NUM_JAC_MIN_FACTOR: Final[float] = ...
NUM_JAC_FACTOR_INCREASE: Final[float] = 10
NUM_JAC_FACTOR_DECREASE: Final[float] = 0.1

class OdeSolution:
    ts: onp.Array1D[np.float64]
    ts_sorted: onp.Array1D[np.float64]
    t_min: np.float64
    t_max: np.float64
    ascending: bool
    side: _Side
    n_segments: int
    interpolants: _Interpolants

    def __init__(self, /, ts: onp.ToFloat1D, interpolants: _Interpolants, alt_segment: op.CanBool = False) -> None: ...
    @overload
    def __call__(self, /, t: float | _ToFloat64) -> onp.Array1D[np.float64]: ...
    @overload
    def __call__(self, /, t: np.complex64 | np.complex128) -> onp.Array1D[np.complex128]: ...
    @overload
    def __call__(self, /, t: np.longdouble) -> onp.Array1D[np.longdouble]: ...
    @overload
    def __call__(self, /, t: np.clongdouble) -> onp.Array1D[np.clongdouble]: ...
    @overload
    def __call__(self, /, t: complex) -> onp.Array1D[np.float64 | np.complex128]: ...
    @overload
    def __call__(self, /, t: Sequence[float | _ToFloat64] | onp.CanArrayND[_ToFloat64]) -> onp.Array2D[np.float64]: ...
    @overload
    def __call__(self, /, t: _To1D[np.complex64 | np.complex128]) -> onp.Array2D[np.complex128]: ...
    @overload
    def __call__(self, /, t: _To1D[np.clongdouble]) -> onp.Array2D[np.clongdouble]: ...
    @overload
    def __call__(self, /, t: Sequence[complex]) -> onp.Array2D[np.float64 | np.complex128]: ...

def validate_first_step(first_step: _ToFloatT, t0: onp.ToFloat, t_bound: onp.ToFloat) -> _ToFloatT: ...
def validate_max_step(max_step: _ToFloatT) -> _ToFloatT: ...
def warn_extraneous(extraneous: dict[str, object]) -> None: ...
def validate_tol(
    rtol: onp.ArrayND[np.floating[Any]],
    atol: onp.ArrayND[np.floating[Any]],
    n: int,
) -> tuple[onp.Array1D[np.floating[Any]], onp.Array1D[np.floating[Any]]]: ...
def norm(x: onp.ToFloatND) -> np.floating[Any]: ...
def select_initial_step(
    fun: Callable[[np.float64, onp.Array1D[np.float64]], onp.Array1D[np.float64]],
    t0: float | np.float64,
    y0: onp.ArrayND[np.float64],
    t_bound: float | np.float64,
    max_step: float | np.float64,
    f0: onp.ArrayND[np.float64],
    direction: float | np.float64,
    order: float | np.float64,
    rtol: float | np.float64,
    atol: float | np.float64,
) -> float | np.float64: ...
def num_jac(
    fun: Callable[[np.float64, onp.Array1D[np.float64]], onp.Array1D[np.float64]],
    t: float | np.float64,
    y: onp.ArrayND[np.float64],
    f: onp.ArrayND[np.float64],
    threshold: float | np.float64,
    factor: onp.ArrayND[np.float64] | None,
    sparsity: tuple[csc_matrix, onp.ArrayND[np.intp]] | None = None,
) -> tuple[onp.Array2D[np.float64] | csc_matrix, onp.Array1D[np.float64]]: ...
