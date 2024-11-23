from typing import Any, Literal, TypeAlias

import numpy as np
import optype.numpy as onp
from ._ltisys import lti

__all__ = ["abcd_normalize", "cont2discrete", "ss2tf", "ss2zpk", "tf2ss", "zpk2ss"]

_ToSystemTF: TypeAlias = tuple[onp.ToComplex2D, onp.ToComplex1D]  # (num, den)
_InSystemZPK: TypeAlias = tuple[onp.ToComplex1D, onp.ToComplex1D, onp.ToFloat]  # (z, p, k)
_InSystemSS: TypeAlias = tuple[onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D, onp.ToComplex2D]  # (A, B, C, D)

_Inexact1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_SystemTF: TypeAlias = tuple[_Inexact2D, _Inexact1D]
_SystemZPK: TypeAlias = tuple[_Inexact1D, _Inexact1D, float]
_SystemSS: TypeAlias = tuple[_Inexact2D, _Inexact2D, _Inexact2D, _Inexact2D]

_Method: TypeAlias = Literal["gbt", "bilinear", "euler", "backward_diff", "foh", "impulse", "zoh"]

def abcd_normalize(
    A: onp.ToComplex2D | None = None,
    B: onp.ToComplex2D | None = None,
    C: onp.ToComplex2D | None = None,
    D: onp.ToComplex2D | None = None,
) -> _SystemTF: ...
def tf2ss(num: onp.ToComplex2D, den: onp.ToComplex1D) -> _SystemSS: ...
def zpk2ss(z: onp.ToComplex1D, p: onp.ToComplex1D, k: onp.ToFloat) -> _SystemSS: ...
def ss2tf(A: onp.ToComplex2D, B: onp.ToComplex2D, C: onp.ToComplex2D, D: onp.ToComplex2D, input: onp.ToInt = 0) -> _SystemTF: ...
def ss2zpk(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    C: onp.ToComplex2D,
    D: onp.ToComplex2D,
    input: onp.ToInt = 0,
) -> _SystemZPK: ...
def cont2discrete(
    system: lti | _ToSystemTF | _InSystemZPK | _InSystemSS,
    dt: float,
    method: _Method = "zoh",
    alpha: onp.ToFloat | None = None,
) -> (
    tuple[_Inexact2D, _Inexact1D, float]
    | tuple[_Inexact1D, _Inexact1D, float, float]
    | tuple[_Inexact2D, _Inexact2D, _Inexact2D, _Inexact2D, float]
): ...
