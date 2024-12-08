from collections.abc import Sequence
from typing import Literal, TypeAlias, overload
from typing_extensions import LiteralString

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped

__all__ = ["bisplev", "bisplrep", "insert", "spalde", "splantider", "splder", "splev", "splint", "splprep", "splrep", "sproot"]

_Falsy: TypeAlias = Literal[False, 0]
_Truthy: TypeAlias = Literal[True, 1]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]

_Task: TypeAlias = Literal[-1, 0, 1]
_Ext: TypeAlias = Literal[0, 1, 2, 3]

_ToTCK: TypeAlias = Sequence[onp.ToFloat1D | onp.ToFloat2D | int]
# `(t, c, k)`
_OutTCK: TypeAlias = tuple[_Float1D, _Float1D, int]
# `([t, c, k], u)`
_OutTCKU: TypeAlias = tuple[Sequence[_Float1D | list[_Float1D] | int], _Float1D]

###

# NOTE: The docs are incorrect about the return type of `splgrep`
@overload  # full_output: falsy = ...
def splprep(
    x: onp.ToFloat2D,
    w: onp.ToFloat1D | None = None,
    u: onp.ToFloat1D | None = None,
    ub: onp.ToFloat | None = None,
    ue: onp.ToFloat | None = None,
    k: int = 3,
    task: _Task = 0,
    s: onp.ToFloat | None = None,
    t: onp.ToFloat1D | None = None,
    full_output: _Falsy = 0,
    nest: int | None = None,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> _OutTCKU: ...
@overload  # full_output: truthy (positional)
def splprep(
    x: onp.ToFloat2D,
    w: onp.ToFloat1D | None,
    u: onp.ToFloat1D | None,
    ub: onp.ToFloat | None,
    ue: onp.ToFloat | None,
    k: int,
    task: _Task,
    s: onp.ToFloat | None,
    t: onp.ToFloat1D | None,
    full_output: _Truthy,
    nest: int | None = None,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> tuple[_OutTCKU, _Float, int, LiteralString]: ...
@overload  # full_output: truthy (keyword)
def splprep(
    x: onp.ToFloat2D,
    w: onp.ToFloat1D | None = None,
    u: onp.ToFloat1D | None = None,
    ub: onp.ToFloat | None = None,
    ue: onp.ToFloat | None = None,
    k: int = 3,
    task: _Task = 0,
    s: onp.ToFloat | None = None,
    t: onp.ToFloat1D | None = None,
    *,
    full_output: _Truthy,
    nest: int | None = None,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> tuple[_OutTCKU, _Float, int, LiteralString]: ...

#
@overload  # full_output: falsy = ...
def splrep(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    w: onp.ToFloat1D | None = None,
    xb: onp.ToFloat | None = None,
    xe: onp.ToFloat | None = None,
    k: int = 3,
    task: _Task = 0,
    s: onp.ToFloat | None = None,
    t: onp.ToFloat1D | None = None,
    full_output: _Falsy = 0,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> _OutTCK: ...
@overload  # full_output: truthy (positional)
def splrep(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    w: onp.ToFloat1D | None,
    xb: onp.ToFloat | None,
    xe: onp.ToFloat | None,
    k: int,
    task: _Task,
    s: onp.ToFloat | None,
    t: onp.ToFloat1D | None,
    full_output: _Truthy,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> tuple[_OutTCK, _Float, int, LiteralString]: ...
@overload  # full_output: truthy (keyword)
def splrep(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    w: onp.ToFloat1D | None = None,
    xb: onp.ToFloat | None = None,
    xe: onp.ToFloat | None = None,
    k: int = 3,
    task: _Task = 0,
    s: onp.ToFloat | None = None,
    t: onp.ToFloat1D | None = None,
    *,
    full_output: _Truthy,
    per: onp.ToBool = 0,
    quiet: onp.ToBool = 1,
) -> tuple[_OutTCK, _Float, int, LiteralString]: ...

#
def splev(x: onp.ToFloatND, tck: _ToTCK, der: int = 0, ext: _Ext = 0) -> _FloatND: ...

#
@overload  # full_output: falsy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: _ToTCK, full_output: _Falsy = 0) -> _Float | list[_Float]: ...
@overload  # full_output: truthy
def splint(a: onp.ToFloat, b: onp.ToFloat, tck: _ToTCK, full_output: _Truthy) -> tuple[_Float | list[_Float], _Float1D]: ...

#
def sproot(tck: _ToTCK, mest: int = 10) -> _Float1D | list[_Float1D]: ...

#
@overload
def spalde(x: onp.ToFloatStrict1D, tck: _ToTCK) -> _Float1D: ...
@overload
def spalde(x: onp.ToFloatStrict2D, tck: _ToTCK) -> list[_Float1D]: ...
@overload
def spalde(x: onp.ToFloat1D | onp.ToFloat2D, tck: _ToTCK) -> _Float1D | list[_Float1D]: ...

#
# TODO(jorenham): full_output=True
def bisplrep(
    x: Untyped,
    y: Untyped,
    z: Untyped,
    w: Untyped | None = None,
    xb: Untyped | None = None,
    xe: Untyped | None = None,
    yb: Untyped | None = None,
    ye: Untyped | None = None,
    kx: int = 3,
    ky: int = 3,
    task: _Task = 0,
    s: Untyped | None = None,
    eps: float = 1e-16,
    tx: Untyped | None = None,
    ty: Untyped | None = None,
    full_output: _Falsy = 0,
    nxest: Untyped | None = None,
    nyest: Untyped | None = None,
    quiet: int = 1,
) -> Untyped: ...

#
def bisplev(x: Untyped, y: Untyped, tck: Untyped, dx: int = 0, dy: int = 0) -> Untyped: ...

#
def dblint(xa: Untyped, xb: Untyped, ya: Untyped, yb: Untyped, tck: Untyped) -> Untyped: ...

#
def insert(x: Untyped, tck: Untyped, m: int = 1, per: int = 0) -> Untyped: ...

#
def splder(tck: Untyped, n: int = 1) -> Untyped: ...

#
def splantider(tck: Untyped, n: int = 1) -> Untyped: ...
