from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt, _ArrayLikeNumber_co
from optype import CanBool, CanIndex
from scipy._typing import AnyInt, AnyShape

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]
_Plan: TypeAlias = object  # not used by scipy

_ArrayReal: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]
_ArrayComplex: TypeAlias = npt.NDArray[np.complex64 | np.complex128 | np.clongdouble]

###

def fft(
    x: _ArrayLikeNumber_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def fft2(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def fftn(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def ifft(
    x: _ArrayLikeNumber_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ifft2(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ifftn(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def rfft(
    x: _ArrayLikeFloat_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def rfft2(
    x: _ArrayLikeFloat_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def rfftn(
    x: _ArrayLikeFloat_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def irfft(
    x: _ArrayLikeNumber_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def irfft2(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def irfftn(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...

#
def hfft(
    x: _ArrayLikeNumber_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def hfft2(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def hfftn(
    x: _ArrayLikeNumber_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...

#
def ihfft(
    x: _ArrayLikeFloat_co,
    n: AnyInt | None = None,
    axis: CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ihfft2(
    x: _ArrayLikeFloat_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ihfftn(
    x: _ArrayLikeFloat_co,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
