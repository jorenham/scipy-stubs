from scipy._typing import Untyped
from scipy.special._testutils import assert_func_equal as assert_func_equal

class Arg:
    def __init__(self, a=..., b=..., inclusive_a: bool = True, inclusive_b: bool = True): ...
    def values(self, n) -> Untyped: ...

class FixedArg:
    def __init__(self, values) -> None: ...
    def values(self, n) -> Untyped: ...

class ComplexArg:
    real: Untyped
    imag: Untyped
    def __init__(self, a=..., b=...) -> None: ...
    def values(self, n) -> Untyped: ...

class IntArg:
    a: Untyped
    b: Untyped
    def __init__(self, a: int = -1000, b: int = 1000): ...
    def values(self, n) -> Untyped: ...

def get_args(argspec, n) -> Untyped: ...

class MpmathData:
    scipy_func: Untyped
    mpmath_func: Untyped
    arg_spec: Untyped
    dps: Untyped
    prec: Untyped
    n: Untyped
    rtol: Untyped
    atol: Untyped
    ignore_inf_sign: Untyped
    nan_ok: Untyped
    is_complex: Untyped
    distinguish_nan_and_inf: Untyped
    name: Untyped
    param_filter: Untyped
    def __init__(
        self,
        scipy_func,
        mpmath_func,
        arg_spec,
        name: Untyped | None = None,
        dps: Untyped | None = None,
        prec: Untyped | None = None,
        n: Untyped | None = None,
        rtol: float = 1e-07,
        atol: float = 1e-300,
        ignore_inf_sign: bool = False,
        distinguish_nan_and_inf: bool = True,
        nan_ok: bool = True,
        param_filter: Untyped | None = None,
    ): ...
    def check(self) -> Untyped: ...

def assert_mpmath_equal(*a, **kw): ...
def nonfunctional_tooslow(func) -> Untyped: ...
def mpf2float(x) -> Untyped: ...
def mpc2complex(x) -> Untyped: ...
def trace_args(func) -> Untyped: ...

POSIX: Untyped

class TimeoutError(Exception): ...

def time_limited(timeout: float = 0.5, return_val=..., use_sigalrm: bool = True) -> Untyped: ...
def exception_to_nan(func) -> Untyped: ...
def inf_to_nan(func) -> Untyped: ...
def mp_assert_allclose(res, std, atol: int = 0, rtol: float = 1e-17): ...
