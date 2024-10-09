from scipy._typing import Untyped
from ._fitpack_impl import bisplev, bisplrep

__all__ = [
    "bisplev",
    "bisplrep",
    "insert",
    "spalde",
    "splantider",
    "splder",
    "splev",
    "splint",
    "splprep",
    "splrep",
    "sproot",
]

def splprep(
    x: Untyped,
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
    x: Untyped,
    y: Untyped,
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
def splev(x: Untyped, tck: Untyped, der: int = 0, ext: int = 0) -> Untyped: ...
def splint(a: Untyped, b: Untyped, tck: Untyped, full_output: int = 0) -> Untyped: ...
def sproot(tck: Untyped, mest: int = 10) -> Untyped: ...
def spalde(x: Untyped, tck: Untyped) -> Untyped: ...
def insert(x: Untyped, tck: Untyped, m: int = 1, per: int = 0) -> Untyped: ...
def splder(tck: Untyped, n: int = 1) -> Untyped: ...
def splantider(tck: Untyped, n: int = 1) -> Untyped: ...
