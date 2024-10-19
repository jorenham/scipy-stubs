from scipy._typing import Untyped

def fft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfft(
    x: Untyped,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def fftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def fft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ifft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def rfft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def irfft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def hfft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfftn(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
def ihfft2(
    x: Untyped,
    s: Untyped | None = None,
    axes: Untyped = (-2, -1),
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...
