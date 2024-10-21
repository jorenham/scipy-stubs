from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyBool, AnyReal
from scipy.sparse import csc_matrix
from .base import DenseOutput

_Side: TypeAlias = Literal["left", "right"]
_Interpolants: TypeAlias = Sequence[DenseOutput]
_Float64_co: TypeAlias = np.float16 | np.float32 | np.float64 | np.integer[Any] | np.bool_

_SCT = TypeVar("_SCT", bound=np.generic)
_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)
_AnyRealT = TypeVar("_AnyRealT", bound=AnyReal)

@type_check_only
class _CanLenArray(Protocol[_SCT_co]):
    def __len__(self, /) -> int: ...
    def __array__(self, /) -> npt.NDArray[_SCT_co]: ...

_Vector: TypeAlias = onpt.Array[tuple[int], _SCT]
_Matrix: TypeAlias = onpt.Array[tuple[int, int], _SCT]

###

EPS: Final[float] = ...
NUM_JAC_DIFF_REJECT: Final[float] = ...
NUM_JAC_DIFF_SMALL: Final[float] = ...
NUM_JAC_DIFF_BIG: Final[float] = ...
NUM_JAC_MIN_FACTOR: Final[float] = ...
NUM_JAC_FACTOR_INCREASE: Final[float] = 10
NUM_JAC_FACTOR_DECREASE: Final[float] = 0.1

class OdeSolution:
    ts: _Vector[np.float64]
    ts_sorted: _Vector[np.float64]
    t_min: np.float64
    t_max: np.float64
    ascending: bool
    side: _Side
    n_segments: int
    interpolants: _Interpolants

    def __init__(
        self,
        /,
        ts: _ArrayLikeFloat_co,
        interpolants: _Interpolants,
        alt_segment: AnyBool = False,
    ) -> None: ...
    @overload
    def __call__(self, /, t: float | _Float64_co) -> _Vector[np.float64]: ...
    @overload
    def __call__(self, /, t: np.complex64 | np.complex128) -> _Vector[np.complex128]: ...
    @overload
    def __call__(self, /, t: np.longdouble) -> _Vector[np.longdouble]: ...
    @overload
    def __call__(self, /, t: np.clongdouble) -> _Vector[np.clongdouble]: ...
    @overload
    def __call__(self, /, t: complex) -> _Vector[np.float64 | np.complex128]: ...
    @overload
    def __call__(self, /, t: Sequence[float | _Float64_co] | _CanLenArray[_Float64_co]) -> _Matrix[np.float64]: ...
    @overload
    def __call__(
        self,
        /,
        t: Sequence[np.complex64 | np.complex128] | _CanLenArray[np.complex64 | np.complex128],
    ) -> _Matrix[np.complex128]: ...
    @overload
    def __call__(self, /, t: Sequence[np.clongdouble] | _CanLenArray[np.clongdouble]) -> _Matrix[np.clongdouble]: ...
    @overload
    def __call__(self, /, t: Sequence[complex]) -> _Matrix[np.float64 | np.complex128]: ...

def validate_first_step(first_step: _AnyRealT, t0: AnyReal, t_bound: AnyReal) -> _AnyRealT: ...
def validate_max_step(max_step: _AnyRealT) -> _AnyRealT: ...
def warn_extraneous(extraneous: dict[str, object]) -> None: ...
def validate_tol(
    rtol: npt.NDArray[np.floating[Any]],
    atol: npt.NDArray[np.floating[Any]],
    n: int,
) -> tuple[_Vector[np.floating[Any]], _Vector[np.floating[Any]]]: ...
def norm(x: npt.ArrayLike) -> np.floating[Any]: ...
def select_initial_step(
    fun: Callable[[np.float64, _Vector[np.float64]], _Vector[np.float64]],
    t0: float | np.float64,
    y0: npt.NDArray[np.float64],
    t_bound: float | np.float64,
    max_step: float | np.float64,
    f0: npt.NDArray[np.float64],
    direction: float | np.float64,
    order: float | np.float64,
    rtol: float | np.float64,
    atol: float | np.float64,
) -> float | np.float64: ...
def num_jac(
    fun: Callable[[np.float64, _Vector[np.float64]], _Vector[np.float64]],
    t: float | np.float64,
    y: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    threshold: float | np.float64,
    factor: npt.NDArray[np.float64] | None,
    sparsity: tuple[csc_matrix, npt.NDArray[np.intp]] | None = None,
) -> tuple[_Matrix[np.float64] | csc_matrix, _Vector[np.float64]]: ...
