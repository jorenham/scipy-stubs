from numpy._typing import _ArrayLikeInt, _ArrayLikeNumber_co
from optype import CanBool
from scipy._typing import AnyInt, AnyShape, DCTType, NormalizationMode
from ._realtransforms import _ArrayReal, dct as dct, dctn as dctn, dst as dst, dstn as dstn, idct as idct, idst as idst

# NOTE: Unlike the ones in `scipy.fft._realtransforms`, `orthogonalize` is keyword-only here.
def idctn(
    x: _ArrayLikeNumber_co,
    type: DCTType = 2,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    orthogonalize: CanBool | None = None,
) -> _ArrayReal: ...
def idstn(
    x: _ArrayLikeNumber_co,
    type: DCTType = 2,
    s: _ArrayLikeInt | None = None,
    axes: AnyShape | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: CanBool = False,
    workers: AnyInt | None = None,
    *,
    orthogonalize: CanBool | None = None,
) -> _ArrayReal: ...
