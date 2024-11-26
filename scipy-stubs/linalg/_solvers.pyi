from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = [
    "solve_continuous_are",
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_lyapunov",
    "solve_sylvester",
]

def solve_sylvester(a: npt.ArrayLike, b: npt.ArrayLike, q: npt.ArrayLike) -> onp.Array2D[np.inexact[Any]]: ...
def solve_continuous_lyapunov(a: npt.ArrayLike, q: npt.ArrayLike) -> onp.Array2D[np.inexact[Any]]: ...

solve_lyapunov = solve_continuous_lyapunov

def solve_discrete_lyapunov(
    a: npt.ArrayLike,
    q: npt.ArrayLike,
    method: Literal["direct", "bilinear"] | None = None,
) -> onp.Array2D[np.inexact[Any]]: ...
def solve_continuous_are(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    q: npt.ArrayLike,
    r: npt.ArrayLike,
    e: npt.ArrayLike | None = None,
    s: npt.ArrayLike | None = None,
    balanced: bool = True,
) -> onp.Array2D[np.inexact[Any]]: ...
def solve_discrete_are(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    q: npt.ArrayLike,
    r: npt.ArrayLike,
    e: npt.ArrayLike | None = None,
    s: npt.ArrayLike | None = None,
    balanced: bool = True,
) -> onp.Array2D[np.inexact[Any]]: ...
