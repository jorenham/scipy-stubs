from scipy._typing import Untyped

def c2c(
    forward,
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

fft: Untyped
ifft: Untyped

def r2c(
    forward,
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

rfft: Untyped
ihfft: Untyped

def c2r(
    forward,
    x,
    n: Untyped | None = None,
    axis: int = -1,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

hfft: Untyped
irfft: Untyped

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
def c2cn(
    forward,
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

fftn: Untyped
ifftn: Untyped

def r2cn(
    forward,
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

rfftn: Untyped
ihfftn: Untyped

def c2rn(
    forward,
    x,
    s: Untyped | None = None,
    axes: Untyped | None = None,
    norm: Untyped | None = None,
    overwrite_x: bool = False,
    workers: Untyped | None = None,
    *,
    plan: Untyped | None = None,
) -> Untyped: ...

hfftn: Untyped
irfftn: Untyped

def r2r_fftpack(
    forward, x, n: Untyped | None = None, axis: int = -1, norm: Untyped | None = None, overwrite_x: bool = False
) -> Untyped: ...

rfft_fftpack: Untyped
irfft_fftpack: Untyped
