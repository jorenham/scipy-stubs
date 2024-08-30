from scipy._typing import Untyped

def fft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfft(
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def fftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def fft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfftn(
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfft2(
    x,
    s: Untyped | None = None,
    axes=(-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
