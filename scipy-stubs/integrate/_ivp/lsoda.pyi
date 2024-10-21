from collections.abc import Callable
from typing_extensions import Never

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from .base import DenseOutput, OdeSolver

class LSODA(OdeSolver):
    def __init__(
        self,
        /,
        fun: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        t0: float,
        y0: npt.NDArray[np.float64],
        t_bound: float,
        first_step: float | None = None,
        min_step: float = 0.0,
        max_step: float = ...,
        rtol: _ArrayLikeFloat_co = 0.001,
        atol: _ArrayLikeFloat_co = 1e-06,
        jac: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
        lband: int | None = None,
        uband: int | None = None,
        vectorized: bool = False,
        **extraneous: Never,
    ) -> None: ...

class LsodaDenseOutput(DenseOutput):
    h: float
    yh: npt.NDArray[np.float64]
    p: npt.NDArray[np.intp]

    def __init__(self, /, t_old: float, t: float, h: float, order: int, yh: npt.NDArray[np.float64]) -> None: ...
