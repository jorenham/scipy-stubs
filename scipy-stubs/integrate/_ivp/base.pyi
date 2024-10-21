from collections.abc import Callable
from typing import Any, ClassVar, Final, Generic, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._typing import AnyReal

_VT = TypeVar("_VT", bound=npt.NDArray[np.inexact[Any]], default=npt.NDArray[np.inexact[Any]])

class OdeSolver:
    TOO_SMALL_STEP: ClassVar[str] = ...

    t: float
    t_old: float
    t_bound: float
    vectorized: bool
    fun: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    fun_single: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    fun_vectorized: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    direction: float
    n: int
    status: Literal["running", "finished", "failed"]
    nfev: int
    njev: int
    nlu: int

    @overload
    def __init__(
        self,
        fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeFloat_co],
        t0: AnyReal,
        y0: _ArrayLikeFloat_co,
        t_bound: AnyReal,
        vectorized: bool,
        support_complex: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        fun: Callable[[float, npt.NDArray[np.float64 | np.complex128]], _ArrayLikeNumber_co],
        t0: AnyReal,
        y0: _ArrayLikeNumber_co,
        t_bound: AnyReal,
        vectorized: bool,
        support_complex: Literal[True],
    ) -> None: ...
    @property
    def step_size(self) -> float | None: ...
    def step(self) -> str | None: ...
    def dense_output(self) -> ConstantDenseOutput: ...

class DenseOutput:
    t_old: Final[float]
    t: Final[float]
    t_min: Final[float]
    t_max: Final[float]

    def __init__(self, /, t_old: AnyReal, t: AnyReal) -> None: ...
    @overload
    def __call__(self, /, t: AnyReal) -> onpt.Array[tuple[int], np.inexact[Any]]: ...
    @overload
    def __call__(self, /, t: _ArrayLikeFloat_co) -> npt.NDArray[np.inexact[Any]]: ...

class ConstantDenseOutput(DenseOutput, Generic[_VT]):
    value: _VT
    def __init__(self, /, t_old: AnyReal, t: AnyReal, value: _VT) -> None: ...

def check_arguments(
    fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeNumber_co],
    y0: _ArrayLikeNumber_co,
    support_complex: bool,
) -> (
    Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    | Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.complex128]]
): ...
