from scipy._typing import Untyped

dfitpack_int: Untyped

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
def bisplrep(
    x,
    y,
    z,
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
def bisplev(x, y, tck, dx: int = 0, dy: int = 0) -> Untyped: ...
def dblint(xa, xb, ya, yb, tck) -> Untyped: ...
def insert(x, tck, m: int = 1, per: int = 0) -> Untyped: ...
def splder(tck, n: int = 1) -> Untyped: ...
def splantider(tck, n: int = 1) -> Untyped: ...
