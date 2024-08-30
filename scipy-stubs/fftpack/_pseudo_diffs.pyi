from scipy._typing import Untyped

__all__ = ["cc_diff", "cs_diff", "diff", "hilbert", "ihilbert", "itilbert", "sc_diff", "shift", "ss_diff", "tilbert"]

def diff(x, order: int = 1, period: Untyped | None = None, _cache=...) -> Untyped: ...
def tilbert(x, h, period: Untyped | None = None, _cache=...) -> Untyped: ...
def itilbert(x, h, period: Untyped | None = None, _cache=...) -> Untyped: ...
def hilbert(x, _cache=...) -> Untyped: ...
def ihilbert(x) -> Untyped: ...
def cs_diff(x, a, b, period: Untyped | None = None, _cache=...) -> Untyped: ...
def sc_diff(x, a, b, period: Untyped | None = None, _cache=...) -> Untyped: ...
def ss_diff(x, a, b, period: Untyped | None = None, _cache=...) -> Untyped: ...
def cc_diff(x, a, b, period: Untyped | None = None, _cache=...) -> Untyped: ...
def shift(x, a, period: Untyped | None = None, _cache=...) -> Untyped: ...
