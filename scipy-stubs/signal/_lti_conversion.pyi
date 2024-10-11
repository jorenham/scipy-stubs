from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import numpy as np
from numpy._typing import _ArrayLikeComplex_co
from scipy._typing import AnyInt, AnyReal
from ._ltisys import lti

__all__ = ["abcd_normalize", "cont2discrete", "ss2tf", "ss2zpk", "tf2ss", "zpk2ss"]

_InexactT = TypeVar("_InexactT", bound=np.inexact[Any], default=np.inexact[Any])

_InVector: TypeAlias = _ArrayLikeComplex_co
_InMatrix: TypeAlias = _ArrayLikeComplex_co
_InSystemTF: TypeAlias = tuple[_ArrayLikeComplex_co, _ArrayLikeComplex_co]
_InSystemZPK: TypeAlias = tuple[_ArrayLikeComplex_co, _ArrayLikeComplex_co, AnyReal]
_InSystemSS: TypeAlias = tuple[_ArrayLikeComplex_co, _ArrayLikeComplex_co, _ArrayLikeComplex_co, _ArrayLikeComplex_co]

_OutVector: TypeAlias = np.ndarray[tuple[int], np.dtype[_InexactT]]
_OutMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_InexactT]]
_OutSystemTF: TypeAlias = tuple[_OutMatrix[_InexactT], _OutVector[_InexactT]]
_OutSystemZPK: TypeAlias = tuple[_OutVector[_InexactT], _OutVector[_InexactT], float | _InexactT]
_OutSystemSS: TypeAlias = tuple[_OutMatrix[_InexactT], _OutMatrix[_InexactT], _OutMatrix[_InexactT], _OutMatrix[_InexactT]]

_Method: TypeAlias = Literal["gbt", "bilinear", "euler", "backward_diff", "foh", "impulse", "zoh"]

def abcd_normalize(
    A: _InMatrix | None = None,
    B: _InMatrix | None = None,
    C: _InMatrix | None = None,
    D: _InMatrix | None = None,
) -> _OutSystemTF: ...
def tf2ss(num: _InMatrix, den: _InVector) -> _OutSystemSS: ...
def zpk2ss(z: _InVector, p: _InVector, k: AnyReal) -> _OutSystemSS: ...
def ss2zpk(A: _InMatrix, B: _InMatrix, C: _InMatrix, D: _InMatrix, input: AnyInt = 0) -> _OutSystemZPK: ...
def ss2tf(A: _InMatrix, B: _InMatrix, C: _InMatrix, D: _InMatrix, input: AnyInt = 0) -> _OutSystemTF: ...
def cont2discrete(
    system: lti | _InSystemTF | _InSystemZPK | _InSystemSS,
    dt: float,
    method: _Method = "zoh",
    alpha: AnyReal | None = None,
) -> (
    tuple[_OutMatrix, _OutVector, float]
    | tuple[_OutVector, _OutVector, float, float]
    | tuple[_OutMatrix, _OutMatrix, _OutMatrix, _OutMatrix, float]
): ...
