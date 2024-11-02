from typing import Protocol, TypeAlias, type_check_only

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from optype import CanIndex
from scipy._typing import AnyBool, AnyShape

__all__ = ["fft", "fft2", "fftn", "ifft", "ifft2", "ifftn", "irfft", "rfft"]

_ArrayReal: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]  # no float16
_ArrayComplex: TypeAlias = npt.NDArray[np.complex64 | np.complex128 | np.clongdouble]

@type_check_only
class _OrderedIndex(CanIndex, Protocol):
    def __lt__(self, other: CanIndex, /) -> bool: ...
    def __le__(self, other: CanIndex, /) -> bool: ...

###

def fft(
    x: _ArrayLikeNumber_co,
    n: _OrderedIndex | None = None,
    axis: CanIndex = -1,
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
def ifft(
    x: _ArrayLikeNumber_co,
    n: _OrderedIndex | None = None,
    axis: CanIndex = -1,
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
def rfft(
    x: _ArrayLikeFloat_co,
    n: _OrderedIndex | None = None,
    axis: CanIndex = -1,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
def irfft(
    x: _ArrayLikeFloat_co,
    n: _OrderedIndex | None = None,
    axis: CanIndex = -1,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
def fftn(
    x: _ArrayLikeNumber_co,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
def ifftn(
    x: _ArrayLikeNumber_co,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
def fft2(
    x: _ArrayLikeNumber_co,
    shape: AnyShape | None = None,
    axes: tuple[CanIndex, CanIndex] = (-2, -1),
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
def ifft2(
    x: _ArrayLikeNumber_co,
    shape: AnyShape | None = None,
    axes: tuple[CanIndex, CanIndex] = (-2, -1),
    overwrite_x: AnyBool = False,
) -> _ArrayComplex: ...
