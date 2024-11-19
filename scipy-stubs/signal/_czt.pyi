from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeComplex_co

__all__ = ["CZT", "ZoomFFT", "czt", "czt_points", "zoom_fft"]

class CZT:
    def __init__(self, n: int, m: int | None = None, w: onp.ToComplex | None = None, a: onp.ToComplex = 1 + 0j) -> None: ...
    def __call__(self, x: _ArrayLikeComplex_co, *, axis: int = -1) -> npt.NDArray[np.complex128 | np.complex64]: ...
    def points(self) -> npt.NDArray[np.complex128 | np.complex64]: ...

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
        n: int,
        fn: Sequence[onp.ToFloat] | onp.ToFloat,
        m: int | None = None,
        *,
        fs: onp.ToFloat = 2,
        endpoint: bool = False,
    ) -> None: ...

def czt_points(m: int, w: onp.ToComplex | None = None, a: onp.ToComplex = ...) -> npt.NDArray[np.complex128 | np.complex64]: ...
def czt(
    x: _ArrayLikeComplex_co,
    m: int | None = None,
    w: onp.ToComplex | None = None,
    a: onp.ToComplex = 1 + 0j,
    *,
    axis: int = -1,
) -> npt.NDArray[np.complex128 | np.complex64]: ...
def zoom_fft(
    x: _ArrayLikeComplex_co,
    fn: Sequence[onp.ToFloat] | onp.ToFloat,
    m: int | None = None,
    *,
    fs: int = 2,
    endpoint: bool = False,
    axis: int = -1,
) -> npt.NDArray[np.float64 | np.float32 | np.complex128 | np.complex64]: ...
