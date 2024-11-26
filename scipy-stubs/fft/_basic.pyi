from typing import Literal, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import AnyShape

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]
_Plan: TypeAlias = object  # not used by scipy

_ArrayReal: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]
_ArrayComplex: TypeAlias = onp.ArrayND[np.complex64 | np.complex128 | np.clongdouble]

###

def fft(
    x: onp.ToComplexND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def fft2(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def fftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def ifft(
    x: onp.ToComplexND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ifft2(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ifftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def rfft(
    x: onp.ToFloatND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def rfft2(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def rfftn(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...

#
def irfft(
    x: onp.ToComplexND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def irfft2(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def irfftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...

#
def hfft(
    x: onp.ToComplexND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def hfft2(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...
def hfftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayReal: ...

#
def ihfft(
    x: onp.ToFloatND,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ihfft2(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
def ihfftn(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    plan: _Plan | None = None,
) -> _ArrayComplex: ...
