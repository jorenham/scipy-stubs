from scipy._typing import DCTType, NormalizationMode, Untyped
from ._realtransforms import (
    dct as dct,
    dctn as dctn,
    dst as dst,
    dstn as dstn,
    idct as idct,
    idst as idst,
)

# NOTE: Unlike the ones in `scipy.fft._realtransforms`, `orthogonalize` is keyword-only here.
def idctn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    *,
    orthogonalize: bool | None = None,
) -> Untyped: ...
def idstn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    *,
    orthogonalize: bool | None = None,
) -> Untyped: ...
