from scipy._typing import Untyped

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

def sawtooth(t: Untyped, width: int = 1) -> Untyped: ...
def square(t: Untyped, duty: float = 0.5) -> Untyped: ...
def gausspulse(
    t: Untyped,
    fc: int = 1000,
    bw: float = 0.5,
    bwr: int = -6,
    tpr: int = -60,
    retquad: bool = False,
    retenv: bool = False,
) -> Untyped: ...
def chirp(
    t: Untyped,
    f0: Untyped,
    t1: Untyped,
    f1: Untyped,
    method: str = "linear",
    phi: int = 0,
    vertex_zero: bool = True,
) -> Untyped: ...
def sweep_poly(t: Untyped, poly: Untyped, phi: int = 0) -> Untyped: ...
def unit_impulse(shape: Untyped, idx: Untyped | None = None, dtype: Untyped = ...) -> Untyped: ...
