from scipy._typing import Untyped

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

dfitpack_int: Untyped

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
    task: int = 0,
    s: Untyped | None = None,
    eps: float = 1e-16,
    tx: Untyped | None = None,
    ty: Untyped | None = None,
    full_output: int = 0,
    nxest: Untyped | None = None,
    nyest: Untyped | None = None,
    quiet: int = 1,
) -> Untyped: ...
def bisplev(x: Untyped, y: Untyped, tck: Untyped, dx: int = 0, dy: int = 0) -> Untyped: ...
def dblint(xa: Untyped, xb: Untyped, ya: Untyped, yb: Untyped, tck: Untyped) -> Untyped: ...
def insert(x: Untyped, tck: Untyped, m: int = 1, per: int = 0) -> Untyped: ...
def splder(tck: Untyped, n: int = 1) -> Untyped: ...
def splantider(tck: Untyped, n: int = 1) -> Untyped: ...
