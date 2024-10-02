from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy._typing as spt
from scipy.optimize import OptimizeResult

def compute_kkt_optimality(g: npt.NDArray[np.floating[Any]] | spt.AnyReal, on_bound: spt.AnyReal) -> np.floating[Any] | float: ...

# TODO(jorenham): custom `OptimizeResult` return type
def bvls(
    A: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    x_lsq: npt.NDArray[np.floating[Any]],
    lb: npt.NDArray[np.floating[Any]],
    ub: npt.NDArray[np.floating[Any]],
    tol: spt.AnyReal,
    max_iter: spt.AnyInt,
    verbose: Literal[0, 1, 2],
    rcond: spt.AnyReal | None = None,
) -> OptimizeResult: ...
