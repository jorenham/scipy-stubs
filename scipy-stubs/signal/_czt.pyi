from typing import TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = ["CZT", "ZoomFFT", "czt", "czt_points", "zoom_fft"]

_Float: TypeAlias = np.float64 | np.float32
_Complex: TypeAlias = np.complex128 | np.complex64

class CZT:
    def __init__(self, /, n: int, m: int | None = None, w: onp.ToComplex | None = None, a: onp.ToComplex = 1 + 0j) -> None: ...
    def __call__(self, /, x: onp.ToComplexND, *, axis: int = -1) -> onp.ArrayND[_Complex]: ...
    def points(self, /) -> onp.ArrayND[_Complex]: ...

class ZoomFFT(CZT):
    w: complex
    a: complex

    m: int
    n: int

    f1: onp.ToFloat
    f2: onp.ToFloat
    fs: onp.ToFloat

    def __init__(
        self,
        /,
        n: int,
        fn: onp.ToFloat | onp.ToFloatND,
        m: int | None = None,
        *,
        fs: onp.ToFloat = 2,
        endpoint: bool = False,
    ) -> None: ...

def czt_points(m: int, w: onp.ToComplex | None = None, a: onp.ToComplex = ...) -> onp.ArrayND[_Complex]: ...
def czt(
    x: onp.ToComplexND,
    m: int | None = None,
    w: onp.ToComplex | None = None,
    a: onp.ToComplex = 1 + 0j,
    *,
    axis: int = -1,
) -> onp.ArrayND[_Complex]: ...
def zoom_fft(
    x: onp.ToComplexND,
    fn: onp.ToFloatND | onp.ToFloat,
    m: int | None = None,
    *,
    fs: int = 2,
    endpoint: bool = False,
    axis: int = -1,
) -> onp.ArrayND[_Float | _Complex]: ...
