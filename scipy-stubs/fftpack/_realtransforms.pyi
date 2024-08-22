from scipy._typing import Untyped

def dctn(
    x,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def idctn(
    x,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def dstn(
    x,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def idstn(
    x,
    type: int = 2,
    shape: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
) -> Untyped: ...
def dct(
    x, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def idct(
    x, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def dst(
    x, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
def idst(
    x, type: int = 2, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...
