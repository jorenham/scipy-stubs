from collections.abc import Callable, Mapping
from typing import Concatenate, Literal, TypeAlias, TypedDict, TypeVar, final, overload, type_check_only
from typing_extensions import Unpack

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse import sparray, spmatrix
from ._constraints import Bounds

__all__ = ["curve_fit", "fixed_point", "fsolve", "leastsq"]

###

_XT = TypeVar("_XT")
_FT = TypeVar("_FT")
_Fun: TypeAlias = Callable[Concatenate[_XT, ...], _FT]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]

_Fun1D: TypeAlias = _Fun[_Float1D, onp.ToFloat1D]
_Fun2D: TypeAlias = _Fun[_Float2D, onp.ToFloat1D]
_Jac1D: TypeAlias = _Fun[_Float1D, onp.ToFloat2D]
_Jac2D: TypeAlias = _Fun[_Float2D, onp.ToFloat2D]

_JacMethod: TypeAlias = Literal["2-point", "3-point", "cs"]
_CurveFitMethod: TypeAlias = Literal["lm", "trf", "dogbox"]
_NanPolicy: TypeAlias = Literal["raise", "omit"]  # no "propagate"
_IERFlag: TypeAlias = Literal[1, 2, 3, 4, 5, 6, 7, 8]

@final
@type_check_only
class _KwargsCurveFit(TypedDict, total=False):
    ftol: onp.ToFloat
    xtol: onp.ToFloat
    gtol: onp.ToFloat

    # leastsq
    col_deriv: onp.ToBool
    maxfev: onp.ToJustInt
    epsfcn: onp.ToFloat | None
    factor: onp.ToJustInt
    diag: onp.ToFloat1D | None

    # least_squares
    x_scale: onp.ToFloat | onp.ToFloatND | Literal["jac"]
    f_scale: onp.ToFloat
    loss: _Fun[_Float1D, onp.ToFloat1D] | Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
    diff_step: onp.ToFloat1D | None
    tr_solver: Literal["exact", "lsmr"]
    tr_options: Mapping[str, object]
    jac_sparsity: onp.ToFloat2D | spmatrix | sparray
    max_nfev: onp.ToJustInt
    verbose: Literal[0, 1, 2]
    kwargs: Mapping[str, object]

@type_check_only
class _InfoDictBase(TypedDict):
    nfev: int
    fvec: _Float1D

@type_check_only
class _InfoDictSolve(_InfoDictBase, TypedDict):
    njev: int
    fjac: _Float2D
    r: _Float1D
    qtf: _Float1D

@type_check_only
class _InfoDictLSQ(_InfoDictBase, TypedDict):
    fjac: _Float2D
    ipvt: onp.Array1D[np.int32]
    qtf: _Float1D

_InfoDictCurveFit: TypeAlias = _InfoDictBase | _InfoDictLSQ

###

#
@overload  # full_output=False (default)
def fsolve(
    func: _Fun1D,
    x0: onp.ToFloat | onp.ToFloat1D,
    args: tuple[object, ...] = (),
    fprime: _Jac1D | None = None,
    full_output: Falsy = 0,
    col_deriv: onp.ToBool = 0,
    xtol: onp.ToFloat = 1.49012e-8,
    maxfev: onp.ToJustInt = 0,
    band: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> _Float1D: ...
@overload  # full_output=True (positional)
def fsolve(
    func: _Fun1D,
    x0: onp.ToFloat | onp.ToFloat1D,
    args: tuple[object, ...],
    fprime: _Jac1D | None,
    full_output: Truthy,
    col_deriv: onp.ToBool = 0,
    xtol: onp.ToFloat = 1.49012e-8,
    maxfev: onp.ToJustInt = 0,
    band: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> tuple[_Float1D, _InfoDictSolve, _IERFlag, str]: ...
@overload  # full_output=True (keyword)
def fsolve(
    func: _Fun1D,
    x0: onp.ToFloat | onp.ToFloat1D,
    args: tuple[object, ...] = (),
    fprime: _Jac1D | None = None,
    *,
    full_output: Truthy,
    col_deriv: onp.ToBool = 0,
    xtol: onp.ToFloat = 1.49012e-8,
    maxfev: onp.ToJustInt = 0,
    band: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> tuple[_Float1D, _InfoDictSolve, _IERFlag, str]: ...

#
@overload  # full_output=False (default)
def leastsq(
    func: _Fun1D,
    x0: onp.ToFloat1D,
    args: tuple[object, ...] = (),
    Dfun: _Jac1D | None = None,
    full_output: Falsy = False,
    col_deriv: onp.ToBool = False,
    ftol: onp.ToFloat = 1.49012e-8,
    xtol: onp.ToFloat = 1.49012e-8,
    gtol: onp.ToFloat = 0.0,
    maxfev: onp.ToJustInt = 0,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> tuple[_Float1D, _IERFlag]: ...
@overload  # full_output=True (positional)
def leastsq(
    func: _Fun1D,
    x0: onp.ToFloat1D,
    args: tuple[object, ...],
    Dfun: _Jac1D | None,
    full_output: Truthy,
    col_deriv: onp.ToBool = False,
    ftol: onp.ToFloat = 1.49012e-8,
    xtol: onp.ToFloat = 1.49012e-8,
    gtol: onp.ToFloat = 0.0,
    maxfev: onp.ToJustInt = 0,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> tuple[_Float1D, _Float2D, _InfoDictLSQ, str, _IERFlag]: ...
@overload  # full_output=True (keyword)
def leastsq(
    func: _Fun1D,
    x0: onp.ToFloat1D,
    args: tuple[object, ...] = (),
    Dfun: _Jac1D | None = None,
    *,
    full_output: Truthy,
    col_deriv: onp.ToBool = False,
    ftol: onp.ToFloat = 1.49012e-8,
    xtol: onp.ToFloat = 1.49012e-8,
    gtol: onp.ToFloat = 0.0,
    maxfev: onp.ToJustInt = 0,
    epsfcn: onp.ToFloat | None = None,
    factor: onp.ToJustInt = 100,
    diag: onp.ToFloat1D | None = None,
) -> tuple[_Float1D, _Float2D, _InfoDictLSQ, str, _IERFlag]: ...

#
@overload  # 1-d `x`, full-output=False
def curve_fit(
    f: _Fun1D,
    xdata: onp.ToFloatStrict1D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac1D | _JacMethod | None = None,
    *,
    full_output: Falsy = False,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float1D, _Float2D]: ...
@overload  # 1-d `x`, full-output=True
def curve_fit(
    f: _Fun1D,
    xdata: onp.ToFloatStrict1D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac1D | _JacMethod | None = None,
    *,
    full_output: Truthy,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float1D, _Float2D, _InfoDictCurveFit, str, _IERFlag]: ...
@overload  # 2-d `x`, full-output=False
def curve_fit(
    f: _Fun2D,
    xdata: onp.ToFloatStrict2D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac2D | _JacMethod | None = None,
    *,
    full_output: Falsy = False,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float2D, _Float2D]: ...
@overload  # 2-d `x`, full-output=True
def curve_fit(
    f: _Fun2D,
    xdata: onp.ToFloatStrict2D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac2D | _JacMethod | None = None,
    *,
    full_output: Truthy,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float2D, _Float2D, _InfoDictCurveFit, str, _IERFlag]: ...
@overload  # ?-d `x`, full-output=False
def curve_fit(
    f: _Fun1D | _Fun2D,
    xdata: onp.ToFloat1D | onp.ToFloat2D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac1D | _Jac2D | _JacMethod | None = None,
    *,
    full_output: Falsy = False,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float1D | _Float2D, _Float2D]: ...
@overload  # ?-d `x`, full-output=True
def curve_fit(
    f: _Fun1D | _Fun2D,
    xdata: onp.ToFloat1D | onp.ToFloat2D,
    ydata: onp.ToFloat1D,
    p0: onp.ToFloat1D | None = None,
    sigma: onp.ToFloat | onp.ToFloat1D | onp.ToFloat2D | None = None,
    absolute_sigma: op.CanBool = False,
    check_finite: op.CanBool | None = None,
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds = ...,
    method: _CurveFitMethod | None = None,
    jac: _Jac1D | _Jac2D | _JacMethod | None = None,
    *,
    full_output: Truthy,
    nan_policy: _NanPolicy | None = None,
    **kwargs: Unpack[_KwargsCurveFit],
) -> tuple[_Float1D | _Float2D, _Float2D, _InfoDictCurveFit, str, _IERFlag]: ...

#
@overload  # 0-d real
def fixed_point(
    func: _Fun[np.float64, onp.ToFloat] | _Fun[float, onp.ToFloat],
    x0: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> np.float64: ...
@overload  # 0-d complex
def fixed_point(
    func: _Fun[np.complex128, onp.ToComplex] | _Fun[complex, onp.ToComplex],
    x0: onp.ToComplex,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> np.float64 | np.complex128: ...
@overload  # 1-d real
def fixed_point(
    func: _Fun[_Float1D, onp.ToFloat1D],
    x0: onp.ToFloat1D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> _Float1D: ...
@overload  # 1-d complex
def fixed_point(
    func: _Fun[onp.Array1D[np.complex128], onp.ToComplex1D],
    x0: onp.ToComplex1D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array1D[np.float64 | np.complex128]: ...
@overload  # 2-d real
def fixed_point(
    func: _Fun[_Float2D, onp.ToFloat2D],
    x0: onp.ToFloat2D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> _Float2D: ...
@overload  # 2-d complex
def fixed_point(
    func: _Fun[onp.Array2D[np.complex128], onp.ToComplex2D],
    x0: onp.ToComplex2D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 3-d real
def fixed_point(
    func: _Fun[onp.Array3D[np.float64], onp.ToFloat3D],
    x0: onp.ToFloat3D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array3D[np.float64]: ...
@overload  # 3-d complex
def fixed_point(
    func: _Fun[onp.Array3D[np.complex128], onp.ToComplex3D],
    x0: onp.ToComplex3D,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # N-d real
def fixed_point(
    func: _Fun[onp.ArrayND[np.float64], onp.ToFloatND],
    x0: onp.ToFloatND,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.ArrayND[np.float64]: ...
@overload  # N-d complex
def fixed_point(
    func: _Fun[onp.ArrayND[np.complex128], onp.ToComplexND],
    x0: onp.ToComplexND,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 1e-08,
    maxiter: onp.ToJustInt = 500,
    method: Literal["del2", "iteration"] = "del2",
) -> onp.ArrayND[np.float64 | np.complex128]: ...
