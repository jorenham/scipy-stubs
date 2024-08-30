from scipy import linalg as linalg
from scipy._typing import Untyped
from ._filter_design import normalize as normalize, tf2zpk as tf2zpk, zpk2tf as zpk2tf

def tf2ss(num, den) -> Untyped: ...
def abcd_normalize(
    A: Untyped | None = None, B: Untyped | None = None, C: Untyped | None = None, D: Untyped | None = None
) -> Untyped: ...
def ss2tf(A, B, C, D, input: int = 0) -> Untyped: ...
def zpk2ss(z, p, k) -> Untyped: ...
def ss2zpk(A, B, C, D, input: int = 0) -> Untyped: ...
def cont2discrete(system, dt, method: str = "zoh", alpha: Untyped | None = None) -> Untyped: ...
