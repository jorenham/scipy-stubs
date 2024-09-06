from collections.abc import Callable
from typing import Concatenate, Literal
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onpt

__all__ = ["ascent", "central_diff_weights", "derivative", "electrocardiogram", "face"]

# this module proofs the existance of an afterlife, because we're currently on scipy 1.14.1 -_-

@deprecated("will be completely removed in SciPy 1.12.0")
def central_diff_weights(Np: int, ndiv: int = 1) -> onpt.Array[tuple[int], np.float64]: ...
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
def ascent() -> onpt.Array[tuple[Literal[512], Literal[512]], np.uint8]: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def face(
    gray: bool = False,
) -> onpt.Array[tuple[Literal[768], Literal[1024]] | tuple[Literal[768], Literal[1024], Literal[3]], np.uint8]: ...
@deprecated("will be completely removed in SciPy 1.12.0")
def electrocardiogram() -> onpt.Array[tuple[Literal[108_000]], np.float64]: ...
