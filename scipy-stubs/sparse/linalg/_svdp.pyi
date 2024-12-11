from typing import Literal

import optype.numpy as onp
from scipy._typing import Seed, UntypedArray
from scipy.sparse import sparray, spmatrix
from ._interface import LinearOperator

__all__ = ["_svdp"]

def _svdp(
    A: onp.ToComplex2D | sparray | spmatrix | LinearOperator,
    k: int,
    which: Literal["LM", "SM"] = "LM",
    irl_mode: bool = True,
    kmax: int | None = None,
    compute_u: bool = True,
    compute_v: bool = True,
    v0: onp.ToComplexND | None = None,
    full_output: bool = False,
    tol: float = 0,
    delta: float | None = None,
    eta: float | None = None,
    anorm: float = 0,
    cgs: bool = False,
    elr: bool = True,
    min_relgap: float | None = 0.002,
    shifts: int | None = None,
    maxiter: int | None = None,
    random_state: Seed | None = None,
) -> UntypedArray | tuple[UntypedArray, ...]: ...
