from scipy import special as special
from scipy._lib._array_api import (
    array_namespace as array_namespace,
    is_cupy as is_cupy,
    is_numpy as is_numpy,
    is_torch as is_torch,
    xp_copy as xp_copy,
    xp_ravel as xp_ravel,
    xp_real as xp_real,
    xp_take_along_axis as xp_take_along_axis,
)
from scipy._typing import Untyped

def nsum(f, a, b, *, step: int = 1, args=(), log: bool = False, maxterms=..., tolerances: Untyped | None = None) -> Untyped: ...
