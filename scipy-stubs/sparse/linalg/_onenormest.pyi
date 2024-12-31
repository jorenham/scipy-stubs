from typing import TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse._base import _spbase
from ._interface import LinearOperator

__all__ = ["onenormest"]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_ToMatrix: TypeAlias = onp.ToComplex2D | LinearOperator | _spbase

#

@overload  # compute_v: falsy, compute_w: falsy
def onenormest(
    A: _ToMatrix,
    t: int = 2,
    itmax: int = 5,
    compute_v: Falsy = False,
    compute_w: Falsy = False,
) -> np.float64: ...
@overload  # compute_v: falsy, compute_w: truthy  (positional)
def onenormest(
    A: _ToMatrix,
    t: int,
    itmax: int,
    compute_v: Falsy,
    compute_w: Truthy,
) -> tuple[np.float64, _Float1D]: ...
@overload  # compute_v: falsy, compute_w: truthy  (keyword)
def onenormest(
    A: _ToMatrix,
    t: int = 2,
    itmax: int = 5,
    compute_v: Falsy = False,
    *,
    compute_w: Truthy,
) -> tuple[np.float64, _Float1D]: ...
@overload  # compute_v: truthy  (positional), compute_w: falsy
def onenormest(
    A: _ToMatrix,
    t: int,
    itmax: int,
    compute_v: Truthy,
    compute_w: Falsy = False,
) -> tuple[np.float64, _Float1D]: ...
@overload  # compute_v: truthy  (keyword), compute_w: falsy
def onenormest(
    A: _ToMatrix,
    t: int = 2,
    itmax: int = 5,
    *,
    compute_v: Truthy,
    compute_w: Falsy = False,
) -> tuple[np.float64, _Float1D]: ...
@overload  # compute_v: truthy  (positional), compute_w: truthy
def onenormest(
    A: _ToMatrix,
    t: int,
    itmax: int,
    compute_v: Truthy,
    compute_w: Truthy,
) -> tuple[np.float64, _Float1D, _Float1D]: ...
@overload  # compute_v: truthy  (keyword), compute_w: truthy
def onenormest(
    A: _ToMatrix,
    t: int = 2,
    itmax: int = 5,
    *,
    compute_v: Truthy,
    compute_w: Truthy,
) -> tuple[np.float64, _Float1D, _Float1D]: ...
