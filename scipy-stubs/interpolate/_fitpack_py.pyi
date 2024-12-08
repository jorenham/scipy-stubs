from collections.abc import Sequence
from typing import Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from ._bsplines import BSpline
from ._fitpack_impl import bisplev, bisplrep, splprep, splrep

__all__ = ["bisplev", "bisplrep", "insert", "spalde", "splantider", "splder", "splev", "splint", "splprep", "splrep", "sproot"]

_Falsy: TypeAlias = Literal[False, 0]
_Truthy: TypeAlias = Literal[True, 1]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]

_Ext: TypeAlias = Literal[0, 1, 2, 3]
_ToTCK: TypeAlias = Sequence[onp.ToFloat1D | onp.ToFloat2D | int]
_TCK: TypeAlias = tuple[_Float1D, _Float1D, int]

###

# NOTE: The following functions also accept `BSpline` instances, unlike their duals in `_fitpack_impl`.

#
@overload  # tck: BSpline
def splev(x: onp.ToFloatND, tck: BSpline, der: int = 0, ext: _Ext = 0) -> _FloatND: ...
@overload  # tck: (t, c, k)
def splev(x: onp.ToFloatND, tck: _ToTCK, der: int = 0, ext: _Ext = 0) -> _FloatND | list[_FloatND]: ...

#
@overload  # tck: BSpline, full_output: falsy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: BSpline, full_output: _Falsy = 0) -> _Float | _Float1D: ...
@overload  # tck: BSpline, full_output: truthy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: BSpline, full_output: _Truthy) -> tuple[_Float | _Float1D, _Float1D]: ...
@overload  # tck: (t, c, k), full_output: falsy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: _ToTCK, full_output: _Falsy = 0) -> _Float | list[_Float]: ...
@overload  # tck: (t, c, k), full_output: truthy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: _ToTCK, full_output: _Truthy) -> tuple[_Float | list[_Float], _Float1D]: ...

#
@overload  # tck: BSpline
def sproot(tck: BSpline, mest: int = 10) -> _Float1D | _Float2D: ...
@overload  # tck: (t, c, k)
def sproot(tck: _ToTCK, mest: int = 10) -> _Float1D | list[_Float1D]: ...

#
@overload  # x: 1-d
def spalde(x: onp.ToFloatStrict1D, tck: BSpline | _ToTCK) -> _Float1D: ...
@overload  # x: 2-d, tck: BSpline
def spalde(x: onp.ToFloatStrict2D, tck: BSpline) -> _Float2D: ...
@overload  # x: 2-d, tck: (t, c, k)
def spalde(x: onp.ToFloatStrict2D, tck: _ToTCK) -> list[_Float1D]: ...
@overload  # tck: BSpline
def spalde(x: onp.ToFloat1D | onp.ToFloat2D, tck: BSpline) -> _Float1D | _Float2D: ...
@overload  # tck: (t, c, k)
def spalde(x: onp.ToFloat1D | onp.ToFloat2D, tck: _ToTCK) -> _Float1D | list[_Float1D]: ...
@overload  # tck: BSpline
def insert(x: onp.ToFloat, tck: BSpline, m: int = 1, per: onp.ToBool = 0) -> BSpline[np.float64]: ...
@overload  # tck: (t, c, k)
def insert(x: onp.ToFloat, tck: _ToTCK, m: int = 1, per: onp.ToBool = 0) -> _TCK: ...

#
@overload  # tck: BSpline
def splder(tck: BSpline, n: int = 1) -> BSpline[np.float64]: ...
@overload  # tck: (t, c, k)
def splder(tck: _ToTCK, n: int = 1) -> _TCK: ...

#
@overload  # tck: BSpline
def splantider(tck: BSpline, n: int = 1) -> BSpline[np.float64]: ...
@overload  # tck: (t, c, k)
def splantider(tck: _ToTCK, n: int = 1) -> _TCK: ...
