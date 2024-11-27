from collections.abc import Callable
from typing import Concatenate, Literal, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import NanPolicy, Untyped, UntypedCallable

__all__ = ["curve_fit", "fixed_point", "fsolve", "leastsq"]

def fsolve(
    func: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    fprime: UntypedCallable | None = None,
    full_output: int = 0,
    col_deriv: int = 0,
    xtol: float = ...,
    maxfev: int = 0,
    band: Untyped | None = None,
    epsfcn: Untyped | None = None,
    factor: int = 100,
    diag: Untyped | None = None,
) -> Untyped: ...
def leastsq(
    func: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    Dfun: UntypedCallable | None = None,
    full_output: bool = False,
    col_deriv: bool = False,
    ftol: float = ...,
    xtol: float = ...,
    gtol: float = 0.0,
    maxfev: int = 0,
    epsfcn: Untyped | None = None,
    factor: int = 100,
    diag: Untyped | None = None,
) -> Untyped: ...
def curve_fit(
    f: UntypedCallable,
    xdata: Untyped,
    ydata: Untyped,
    p0: Untyped | None = None,
    sigma: Untyped | None = None,
    absolute_sigma: bool = False,
    check_finite: Untyped | None = None,
    bounds: Untyped = ...,
    method: Untyped | None = None,
    jac: Untyped | None = None,
    *,
    full_output: bool = False,
    nan_policy: NanPolicy | None = None,
    **kwargs: tuple[object, ...],
) -> Untyped: ...
def check_gradient(
    fcn: UntypedCallable,
    Dfcn: UntypedCallable,
    x0: Untyped,
    args: tuple[object, ...] = (),
    col_deriv: int = 0,
) -> Untyped: ...

#
@overload  # 0-d real
def fixed_point(
    func: Callable[Concatenate[np.float64, ...], onp.ToFloat] | Callable[Concatenate[float, ...], onp.ToFloat],
    x0: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> np.float64: ...
@overload  # 0-d complex
def fixed_point(
    func: Callable[Concatenate[np.complex128, ...], onp.ToFloat] | Callable[Concatenate[float, ...], onp.ToComplex],
    x0: onp.ToComplex,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> np.float64 | np.complex128: ...
@overload  # 1-d real
def fixed_point(
    func: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat1D],
    x0: onp.ToFloat1D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array1D[np.float64]: ...
@overload  # 1-d complex
def fixed_point(
    func: Callable[Concatenate[onp.Array1D[np.complex128], ...], onp.ToComplex1D],
    x0: onp.ToComplex1D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array1D[np.float64 | np.complex128]: ...
@overload  # 2-d real
def fixed_point(
    func: Callable[Concatenate[onp.Array2D[np.float64], ...], onp.ToFloat2D],
    x0: onp.ToFloat2D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array2D[np.float64]: ...
@overload  # 2-d complex
def fixed_point(
    func: Callable[Concatenate[onp.Array2D[np.complex128], ...], onp.ToComplex2D],
    x0: onp.ToComplex2D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 3-d real
def fixed_point(
    func: Callable[Concatenate[onp.Array3D[np.float64], ...], onp.ToFloat3D],
    x0: onp.ToFloat3D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array3D[np.float64]: ...
@overload  # 3-d complex
def fixed_point(
    func: Callable[Concatenate[onp.Array3D[np.complex128], ...], onp.ToComplex3D],
    x0: onp.ToComplex3D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # N-d real
def fixed_point(
    func: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ToFloatND],
    x0: onp.ToFloatND,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.ArrayND[np.float64]: ...
@overload  # N-d complex
def fixed_point(
    func: Callable[Concatenate[onp.ArrayND[np.complex128], ...], onp.ToComplexND],
    x0: onp.ToComplexND,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.ArrayND[np.float64 | np.complex128]: ...
