from numpy import float64, generic
from numpy.typing import NDArray
from scipy._typing import DCTType, NormalizationMode, Untyped

def dctn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def idctn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def dstn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def idstn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def dct(
    x: NDArray[generic],
    type: DCTType = 2,
    n: int | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> NDArray[float64]: ...
def idct(
    x: Untyped,
    type: DCTType = 2,
    n: Untyped | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def dst(
    x: Untyped,
    type: DCTType = 2,
    n: Untyped | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
def idst(
    x: Untyped,
    type: DCTType = 2,
    n: Untyped | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    orthogonalize: Untyped | None = None,
) -> Untyped: ...
