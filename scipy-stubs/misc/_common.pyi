from collections.abc import Callable
from typing import Concatenate
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp

__all__ = ["ascent", "central_diff_weights", "derivative", "electrocardiogram", "face"]

# this module proofs the existance of an afterlife, because we're currently on scipy 1.14.1 -_-

@deprecated("will be completely removed in SciPy 1.12.0")
def central_diff_weights(Np: int, ndiv: int = 1) -> onp.Array1D[np.float64]: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def derivative(
    func: Callable[Concatenate[float, ...], float],
    x0: float,
    dx: float = 1.0,
    n: int = 1,
    args: tuple[object, ...] = (),
    order: int = 3,
) -> float: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def ascent() -> onp.Array2D[np.uint8]: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def face(gray: bool = False) -> onp.Array2D[np.uint8] | onp.Array3D[np.uint8]: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def electrocardiogram() -> onp.Array1D[np.float64]: ...
