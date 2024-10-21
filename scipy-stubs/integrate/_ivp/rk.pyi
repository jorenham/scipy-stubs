from collections.abc import Callable
from typing import ClassVar, Final, Generic, overload
from typing_extensions import Never, TypeVar

import numpy as np
import numpy.typing as npt
from .base import DenseOutput, OdeSolver

_SCT_fc = TypeVar("_SCT_fc", bound=np.float64 | np.complex128, default=np.float64 | np.complex128)

###

SAFETY: Final = 0.9
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

class RungeKutta(OdeSolver, Generic[_SCT_fc]):
    C: ClassVar[npt.NDArray[np.float64]]
    A: ClassVar[npt.NDArray[np.float64]]
    B: ClassVar[npt.NDArray[np.float64]]
    E: ClassVar[npt.NDArray[np.float64]]
    P: ClassVar[npt.NDArray[np.float64]]
    order: ClassVar[int]
    error_estimator_order: ClassVar[int]
    n_stages: ClassVar[int]

    y_old: npt.NDArray[_SCT_fc] | None
    f: npt.NDArray[_SCT_fc]
    K: npt.NDArray[_SCT_fc]
    max_step: float
    h_abs: float
    error_exponent: float
    h_previous: float | None

    def __init__(
        self,
        /,
        fun: Callable[[float, npt.NDArray[_SCT_fc]], npt.NDArray[_SCT_fc]],
        t0: float,
        y0: npt.NDArray[_SCT_fc],
        t_bound: float,
        max_step: float = ...,
        rtol: float = 0.001,
        atol: float = 1e-06,
        vectorized: bool = False,
        first_step: float | None = None,
        **extraneous: Never,
    ) -> None: ...

class RK23(RungeKutta[_SCT_fc], Generic[_SCT_fc]): ...
class RK45(RungeKutta[_SCT_fc], Generic[_SCT_fc]): ...

class DOP853(RungeKutta[_SCT_fc], Generic[_SCT_fc]):
    E3: ClassVar[npt.NDArray[np.float64]]
    E5: ClassVar[npt.NDArray[np.float64]]
    D: ClassVar[npt.NDArray[np.float64]]
    A_EXTRA: ClassVar[npt.NDArray[np.float64]]
    C_EXTRA: ClassVar[npt.NDArray[np.float64]]

    K_extended: npt.NDArray[_SCT_fc]

class RkDenseOutput(DenseOutput, Generic[_SCT_fc]):
    h: float
    order: int
    Q: npt.NDArray[_SCT_fc]
    y_old: npt.NDArray[_SCT_fc]

    def __init__(self, /, t_old: float, t: float, y_old: npt.NDArray[_SCT_fc], Q: npt.NDArray[_SCT_fc]) -> None: ...

class Dop853DenseOutput(DenseOutput, Generic[_SCT_fc]):
    h: float
    F: npt.NDArray[_SCT_fc]
    y_old: npt.NDArray[_SCT_fc]

    def __init__(self, /, t_old: float, t: float, y_old: npt.NDArray[_SCT_fc], F: npt.NDArray[_SCT_fc]) -> None: ...

@overload
def rk_step(
    fun: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    t: float,
    y: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    h: float,
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    K: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def rk_step(
    fun: Callable[[float, npt.NDArray[np.complex128]], npt.NDArray[np.complex128]],
    t: float,
    y: npt.NDArray[np.complex128],
    f: npt.NDArray[np.complex128],
    h: float,
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    K: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: ...
