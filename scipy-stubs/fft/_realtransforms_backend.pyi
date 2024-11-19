import optype as op
from numpy._typing import _ArrayLikeInt, _ArrayLikeNumber_co
from scipy._typing import AnyInt, AnyShape, DCTType, NormalizationMode
from ._realtransforms import _ArrayReal, dct, dctn, dst, dstn, idct, idst

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

# NOTE: Unlike the ones in `scipy.fft._realtransforms`, `orthogonalize` is keyword-only here.
def idctn(
    x: _ArrayLikeNumber_co,
    type: DCTType = 2,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: AnyInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _ArrayReal: ...
def idstn(
    x: _ArrayLikeNumber_co,
    type: DCTType = 2,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: op.CanBool = False,
    workers: AnyInt | None = None,
    *,
    orthogonalize: op.CanBool | None = None,
) -> _ArrayReal: ...
