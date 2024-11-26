from typing import Literal, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import AnyBool, AnyShape, DCTType

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_NormKind: TypeAlias = Literal["ortho"] | None

_ArrayReal: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]  # no float16
_ArrayComplex: TypeAlias = onp.ArrayND[np.complex64 | np.complex128 | np.clongdouble]

###

#
@overload
def dctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dctn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idctn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idctn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dstn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idstn(
    x: onp.ToFloatND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idstn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    shape: AnyShape | None = None,
    axes: AnyShape | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dct(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idct(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idct(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def dst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def dst(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...

#
@overload
def idst(
    x: onp.ToFloatND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal: ...
@overload
def idst(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex | None = None,
    norm: _NormKind = None,
    overwrite_x: AnyBool = False,
) -> _ArrayReal | _ArrayComplex: ...
