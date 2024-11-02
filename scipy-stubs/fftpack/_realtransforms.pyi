from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeFloat_co
from optype import CanIndex
from scipy._typing import AnyBool, AnyInt, AnyShape, DCTType

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_NormKind: TypeAlias = Literal[None, "ortho"]

_ArrayReal: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]  # no float16
_ArrayComplex: TypeAlias = npt.NDArray[np.complex64 | np.complex128 | np.clongdouble]

###

#
@overload
def dctn(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dctn(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idctn(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idctn(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dstn(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dstn(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idstn(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idstn(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dct(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dct(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idct(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idct(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dst(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dst(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idst(
    x: _ArrayLikeFloat_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idst(
    x: _ArrayLikeComplex_co,
    type: DCTType = 2,
    n: AnyInt | None = None,
    axis: CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...
