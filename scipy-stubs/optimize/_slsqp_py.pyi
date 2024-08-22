from ._constraints import old_bound_to_new as old_bound_to_new
from ._numdiff import approx_derivative as approx_derivative
from ._optimize import OptimizeResult as OptimizeResult
from scipy._lib._array_api import array_namespace as array_namespace, xp_atleast_nd as xp_atleast_nd
from scipy._typing import Untyped
from scipy.optimize._slsqp import slsqp as slsqp

__docformat__: str

def approx_jacobian(x, func, epsilon, *args) -> Untyped: ...
def fmin_slsqp(
    func,
    x0,
    eqcons=(),
    f_eqcons: Untyped | None = None,
    ieqcons=(),
    f_ieqcons: Untyped | None = None,
    bounds=(),
    fprime: Untyped | None = None,
    fprime_eqcons: Untyped | None = None,
    fprime_ieqcons: Untyped | None = None,
    args=(),
    iter: int = 100,
    acc: float = 1e-06,
    iprint: int = 1,
    disp: Untyped | None = None,
    full_output: int = 0,
    epsilon=...,
    callback: Untyped | None = None,
) -> Untyped: ...
