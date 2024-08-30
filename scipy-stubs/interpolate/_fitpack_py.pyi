from scipy._typing import Untyped
from ._bsplines import BSpline as BSpline
from ._fitpack_impl import bisplev as bisplev, bisplrep as bisplrep, dblint as dblint

def splprep(
    x,
    w: Untyped | None = None,
    u: Untyped | None = None,
    ub: Untyped | None = None,
    ue: Untyped | None = None,
    k: int = 3,
    task: int = 0,
    s: Untyped | None = None,
    t: Untyped | None = None,
    full_output: int = 0,
    nest: Untyped | None = None,
    per: int = 0,
    quiet: int = 1,
) -> Untyped: ...
def splrep(
    x,
    y,
    w: Untyped | None = None,
    xb: Untyped | None = None,
    xe: Untyped | None = None,
    k: int = 3,
    task: int = 0,
    s: Untyped | None = None,
    t: Untyped | None = None,
    full_output: int = 0,
    per: int = 0,
    quiet: int = 1,
) -> Untyped: ...
def splev(x, tck, der: int = 0, ext: int = 0) -> Untyped: ...
def splint(a, b, tck, full_output: int = 0) -> Untyped: ...
def sproot(tck, mest: int = 10) -> Untyped: ...
def spalde(x, tck) -> Untyped: ...
def insert(x, tck, m: int = 1, per: int = 0) -> Untyped: ...
def splder(tck, n: int = 1) -> Untyped: ...
def splantider(tck, n: int = 1) -> Untyped: ...
