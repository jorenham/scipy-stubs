from typing import TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import AnyShape, DCTType, NormalizationMode
from ._realtransforms import dct, dctn, dst, dstn, idct, idst

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

_RealND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]

# NOTE: Unlike the ones in `scipy.fft._realtransforms`, `orthogonalize` is keyword-only here.
def idctn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _RealND: ...
def idstn(
    x: onp.ToComplexND,
    type: DCTType = 2,
    s: onp.ToInt | onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: onp.ToInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _RealND: ...
