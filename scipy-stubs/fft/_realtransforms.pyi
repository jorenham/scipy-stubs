import numpy.typing as npt
from numpy._typing import _ArrayLikeNumber_co
from scipy._typing import DCTType, NormalizationMode, Untyped

def dctn(
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
def idctn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> Untyped: ...
def dstn(
    x: Untyped,
    type: DCTType = 2,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
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
    orthogonalize: bool | None = None,
) -> Untyped: ...

# We could use overloads based on the type of x to get more accurate return type
# see https://github.com/jorenham/scipy-stubs/pull/118#discussion_r1807957439
def dct(
    x: _ArrayLikeNumber_co,
    type: DCTType = 2,
    n: int | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> npt.NDArray[Untyped]: ...
def idct(
    x: Untyped,
    type: DCTType = 2,
    n: int | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> Untyped: ...
def dst(
    x: Untyped,
    type: DCTType = 2,
    n: int | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> Untyped: ...
def idst(
    x: Untyped,
    type: DCTType = 2,
    n: int | None = None,
    axis: int = -1,
    norm: NormalizationMode | None = None,
    overwrite_x: bool = False,
    workers: int | None = None,
    orthogonalize: bool | None = None,
) -> Untyped: ...
