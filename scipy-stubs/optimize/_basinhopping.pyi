from collections.abc import Callable, Mapping
from typing import Any, Concatenate, Generic, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Seed
from ._minimize import OptimizeResult as _OptimizeResult

__all__ = ["basinhopping"]

_FT = TypeVar("_FT", bound=onp.ToFloat | onp.ToFloatND)
_FT_contra = TypeVar("_FT_contra", bound=onp.ToFloat | onp.ToFloatND, contravariant=True)
_FT_co = TypeVar(
    "_FT_co",
    bound=float | np.floating[Any] | onp.ArrayND[np.floating[Any]],
    default=float | np.float64 | onp.Array1D[np.float64],
    covariant=True,
)

_CallbackFun: TypeAlias = Callable[[onp.Array1D[np.float64], _FT, bool], bool | None]

@type_check_only
class _AcceptTestFun(Protocol[_FT_contra]):
    def __call__(
        self,
        /,
        *,
        f_new: _FT_contra,
        x_new: onp.ToFloat1D,
        f_old: _FT_contra,
        x_old: onp.ToFloat1D,
    ) -> onp.ToBool | Literal["force accept"]: ...

@type_check_only
class OptimizeResult(_OptimizeResult[_FT_co], Generic[_FT_co]):
    lowest_optimization_result: _OptimizeResult[_FT_co]

###

@overload
def basinhopping(
    func: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat],
    x0: onp.ToFloat1D,
    niter: onp.ToJustInt = 100,
    T: onp.ToFloat = 1.0,
    stepsize: onp.ToFloat = 0.5,
    minimizer_kwargs: Mapping[str, object] | None = None,
    take_step: Callable[[onp.Array1D[np.float64]], onp.ToFloat] | None = None,
    accept_test: _AcceptTestFun[onp.ToFloat] | None = None,
    callback: _CallbackFun[float] | _CallbackFun[np.float64] | None = None,
    interval: onp.ToJustInt = 50,
    disp: onp.ToBool = False,
    niter_success: onp.ToJustInt | None = None,
    seed: Seed | None = None,
    *,
    target_accept_rate: onp.ToFloat = 0.5,
    stepwise_factor: onp.ToFloat = 0.9,
) -> OptimizeResult[float | np.float64]: ...
@overload
def basinhopping(
    func: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat1D],
    x0: onp.ToFloat1D,
    niter: onp.ToJustInt = 100,
    T: onp.ToFloat = 1.0,
    stepsize: onp.ToFloat = 0.5,
    minimizer_kwargs: Mapping[str, object] | None = None,
    take_step: Callable[[onp.Array1D[np.float64]], onp.ToFloat] | None = None,
    accept_test: _AcceptTestFun[onp.ToFloat1D] | None = None,
    callback: _CallbackFun[onp.Array1D[np.float64]] | None = None,
    interval: onp.ToJustInt = 50,
    disp: onp.ToBool = False,
    niter_success: onp.ToJustInt | None = None,
    seed: Seed | None = None,
    *,
    target_accept_rate: onp.ToFloat = 0.5,
    stepwise_factor: onp.ToFloat = 0.9,
) -> OptimizeResult[onp.Array1D[np.float64]]: ...
