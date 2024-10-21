from collections.abc import Callable
from typing import Concatenate, TypeAlias, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyBool, AnyInt

__all__ = ["convolve", "convolve_z", "destroy_convolve_cache", "init_convolution_kernel"]

_VectorF8: TypeAlias = onpt.Array[tuple[int], np.float64]

@deprecated("this doesn't do anything; nothing is cached")
def destroy_convolve_cache() -> None: ...
def convolve(
    inout: _ArrayLikeFloat_co,
    omega: _ArrayLikeFloat_co,
    swap_real_imag: AnyBool = False,
    overwrite_x: AnyBool = False,
) -> _VectorF8: ...
def convolve_z(
    inout: _ArrayLikeFloat_co,
    omega_real: _ArrayLikeFloat_co,
    omega_imag: _ArrayLikeFloat_co,
    overwrite_x: AnyBool = False,
) -> _VectorF8: ...
@overload
def init_convolution_kernel(
    n: AnyInt,
    kernel_func: Callable[[int], float],
    d: AnyInt = 0,
    zero_nyquist: AnyInt | None = None,
    kernel_func_extra_args: tuple[()] = (),
) -> _VectorF8: ...
@overload
def init_convolution_kernel(
    n: AnyInt,
    kernel_func: Callable[Concatenate[int, ...], float],
    d: AnyInt,
    zero_nyquist: AnyInt | None,
    kernel_func_extra_args: tuple[object, ...],
) -> _VectorF8: ...
@overload
def init_convolution_kernel(
    n: AnyInt,
    kernel_func: Callable[Concatenate[int, ...], float],
    d: AnyInt = 0,
    zero_nyquist: AnyInt | None = None,
    *,
    kernel_func_extra_args: tuple[object, ...],
) -> _VectorF8: ...
