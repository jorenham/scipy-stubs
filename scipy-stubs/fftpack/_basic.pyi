from scipy._typing import Untyped

def fft(x, n: Untyped | None = None, axis: int = -1, overwrite_x: bool = False) -> Untyped: ...
def ifft(x, n: Untyped | None = None, axis: int = -1, overwrite_x: bool = False) -> Untyped: ...
def rfft(x, n: Untyped | None = None, axis: int = -1, overwrite_x: bool = False) -> Untyped: ...
def irfft(x, n: Untyped | None = None, axis: int = -1, overwrite_x: bool = False) -> Untyped: ...
def fftn(x, shape: Untyped | None = None, axes: Untyped | None = None, overwrite_x: bool = False) -> Untyped: ...
def ifftn(x, shape: Untyped | None = None, axes: Untyped | None = None, overwrite_x: bool = False) -> Untyped: ...
def fft2(x, shape: Untyped | None = None, axes=(-2, -1), overwrite_x: bool = False) -> Untyped: ...
def ifft2(x, shape: Untyped | None = None, axes=(-2, -1), overwrite_x: bool = False) -> Untyped: ...
