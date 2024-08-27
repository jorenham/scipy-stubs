from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "solve_continuous_are",
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_lyapunov",
    "solve_sylvester",
]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

def solve_sylvester(a: npt.ArrayLike, b: npt.ArrayLike, q: npt.ArrayLike) -> _Array_fc_2d: ...
def solve_continuous_lyapunov(a: npt.ArrayLike, q: npt.ArrayLike) -> _Array_fc_2d: ...

solve_lyapunov = solve_continuous_lyapunov

def solve_discrete_lyapunov(
    a: npt.ArrayLike,
    q: npt.ArrayLike,
    method: Literal["direct", "bilinear"] | None = None,
) -> _Array_fc_2d: ...
def solve_continuous_are(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    q: npt.ArrayLike,
    r: npt.ArrayLike,
    e: npt.ArrayLike | None = None,
    s: npt.ArrayLike | None = None,
    balanced: bool = True,
) -> _Array_fc_2d: ...
def solve_discrete_are(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    q: npt.ArrayLike,
    r: npt.ArrayLike,
    e: npt.ArrayLike | None = None,
    s: npt.ArrayLike | None = None,
    balanced: bool = True,
) -> _Array_fc_2d: ...
