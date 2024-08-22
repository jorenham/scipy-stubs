from scipy._typing import Untyped

class IntegrationWarning(UserWarning): ...

def quad(
    func,
    a,
    b,
    args=(),
    full_output: int = 0,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50,
    points: Untyped | None = None,
    weight: Untyped | None = None,
    wvar: Untyped | None = None,
    wopts: Untyped | None = None,
    maxp1: int = 50,
    limlst: int = 50,
    complex_func: bool = False,
) -> Untyped: ...
def dblquad(func, a, b, gfun, hfun, args=(), epsabs: float = 1.49e-08, epsrel: float = 1.49e-08) -> Untyped: ...
def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs: float = 1.49e-08, epsrel: float = 1.49e-08) -> Untyped: ...
def nquad(func, ranges, args: Untyped | None = None, opts: Untyped | None = None, full_output: bool = False) -> Untyped: ...

class _RangeFunc:
    range_: Untyped
    def __init__(self, range_) -> None: ...
    def __call__(self, *args) -> Untyped: ...

class _OptFunc:
    opt: Untyped
    def __init__(self, opt) -> None: ...
    def __call__(self, *args) -> Untyped: ...

class _NQuad:
    abserr: int
    func: Untyped
    ranges: Untyped
    opts: Untyped
    maxdepth: Untyped
    full_output: Untyped
    out_dict: Untyped
    def __init__(self, func, ranges, opts, full_output) -> None: ...
    def integrate(self, *args, **kwargs) -> Untyped: ...
