from typing import TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import AnyShape, DCTType, NormalizationMode

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

# this doesn't include `numpy.float16`
_FloatND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]

###

# TODO: Add overloads for specific return dtypes, as discussed in:
# https://github.com/scipy/scipy-stubs/pull/118#discussion_r1807957439

def dctn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def idctn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def dstn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def idstn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def dct(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def idct(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def dst(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...

#
def idst(
    x: onp.ToComplexND,
    type: DCTType = 2,
    n: onp.ToInt | None = None,
    axis: op.CanIndex = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    orthogonalize: op.CanBool | None = None,
) -> _FloatND: ...
