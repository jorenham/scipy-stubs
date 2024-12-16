from collections.abc import Callable
from typing import Any

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.sparse import csr_array

__all__ = ["NdBSpline"]

class NdBSpline:
    c: onp.ArrayND[np.float64]
    extrapolate: bool

    @property
    def k(self, /) -> int: ...
    @property
    def t(self, /) -> tuple[onp.Array1D[np.float64]]: ...

    #
    def __init__(
        self,
        /,
        t: tuple[onp.ToFloat1D, ...],
        c: onp.ToFloatND,
        k: op.CanIndex | tuple[op.CanIndex, ...],
        *,
        extrapolate: onp.ToBool | None = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        xi: onp.ToFloatND,
        *,
        nu: onp.ToFloat1D | None = None,
        extrapolate: onp.ToBool | None = None,
    ) -> onp.ArrayND[np.floating[Any]]: ...

    #
    @classmethod
    def design_matrix(
        cls,
        xvals: onp.ToFloat2D,
        t: tuple[onp.ToFloat1D, ...],
        k: op.CanIndex | tuple[op.CanIndex, ...],
        extrapolate: onp.ToBool = True,
    ) -> csr_array[np.float32 | np.float64, tuple[int, int]]: ...

def make_ndbspl(
    points: tuple[onp.ToFloat1D, ...],
    values: onp.ToFloatND,
    k: op.CanIndex | tuple[op.CanIndex, ...] = 3,
    *,
    solver: Callable[..., object] = ...,
    **solver_args: object,
) -> NdBSpline: ...  # undocumented
