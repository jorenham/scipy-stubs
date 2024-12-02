from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

__all__ = ["polar"]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Side: TypeAlias = Literal["left", "right"]

###

@overload
def polar(a: onp.ToFloat2D, side: _Side = "right") -> _Tuple2[onp.Array2D[np.floating[Any]]]: ...
@overload
def polar(a: onp.ToComplex2D, side: _Side = "right") -> _Tuple2[onp.Array2D[np.inexact[Any]]]: ...
