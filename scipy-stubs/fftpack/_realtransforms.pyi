from scipy._typing import Untyped

def dctn(
    x: Untyped,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def idctn(
    x: Untyped,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def dstn(
    x: Untyped,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def idstn(
    x: Untyped,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def dct(
    x: Untyped, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def idct(
    x: Untyped, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def dst(
    x: Untyped, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def idst(
    x: Untyped, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
