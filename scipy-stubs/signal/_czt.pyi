from typing import Any, Final, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = ["CZT", "ZoomFFT", "czt", "czt_points", "zoom_fft"]

_Complex: TypeAlias = np.complex128 | np.clongdouble

###

# TODO: make generic on `_Complex`
class CZT:
    w: Final[onp.ToComplex]
    a: Final[onp.ToComplex]
    m: Final[int | np.integer[Any]]
    n: Final[int | np.integer[Any]]

    def __init__(
        self,
        /,
        n: onp.ToJustInt,
        m: onp.ToJustInt | None = None,
        w: onp.ToComplex | None = None,
        a: onp.ToComplex = 1 + 0j,
    ) -> None: ...
    def __call__(self, /, x: onp.ToComplexND, *, axis: int = -1) -> onp.ArrayND[_Complex]: ...
    def points(self, /) -> onp.Array1D[_Complex]: ...

class ZoomFFT(CZT):
    f1: onp.ToFloat
    f2: onp.ToFloat
    fs: onp.ToFloat

    def __init__(
        self,
        /,
        n: onp.ToJustInt,
        fn: onp.ToFloat | onp.ToFloat1D,
        m: onp.ToJustInt | None = None,
        *,
        fs: onp.ToFloat = 2,
        endpoint: onp.ToBool = False,
    ) -> None: ...

#
def _validate_sizes(n: onp.ToJustInt, m: onp.ToJustInt | None) -> int | np.integer[Any]: ...

#
def czt_points(
    m: onp.ToJustInt,
    w: onp.ToComplex | None = None,
    a: onp.ToComplex = 1 + 0j,
) -> onp.Array1D[_Complex]: ...

#
def czt(
    x: onp.ToComplexND,
    m: onp.ToJustInt | None = None,
    w: onp.ToComplex | None = None,
    a: onp.ToComplex = 1 + 0j,
    *,
    axis: op.CanIndex = -1,
) -> onp.ArrayND[_Complex]: ...

#
def zoom_fft(
    x: onp.ToComplexND,
    fn: onp.ToFloatND | onp.ToFloat,
    m: onp.ToJustInt | None = None,
    *,
    fs: onp.ToFloat = 2,
    endpoint: onp.ToBool = False,
    axis: op.CanIndex = -1,
) -> onp.ArrayND[_Complex]: ...
