from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Final, Generic, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy._typing as spt

_VT = TypeVar("_VT", bound=npt.NDArray[np.inexact[Any]], default=npt.NDArray[np.inexact[Any]])

_ArrayLikeReal: TypeAlias = float | Sequence[float] | onpt.AnyFloatingArray | onpt.AnyIntegerArray
_ArrayLikeComplex: TypeAlias = complex | Sequence[complex] | onpt.AnyComplexFloatingArray

def check_arguments(
    fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeComplex],
    y0: _ArrayLikeComplex,
    support_complex: bool,
) -> (
    Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    | Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.complex128]]
): ...

class OdeSolver:
    TOO_SMALL_STEP: ClassVar[str]
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
    def __init__(
        self,
        fun: Callable[[float, npt.NDArray[np.float64]], _ArrayLikeComplex],
        t0: float,
        y0: _ArrayLikeComplex,
        t_bound: float,
        vectorized: bool,
        support_complex: bool = False,
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
    def __init__(self, /, t_old: float, t: float) -> None: ...
    @overload
    def __call__(self, /, t: spt.AnyReal) -> onpt.Array[tuple[int], np.inexact[Any]]: ...
    @overload
    def __call__(self, /, t: _ArrayLikeReal) -> npt.NDArray[np.inexact[Any]]: ...

class ConstantDenseOutput(DenseOutput, Generic[_VT]):
    value: _VT
    def __init__(self, /, t_old: float, t: float, value: _VT) -> None: ...
