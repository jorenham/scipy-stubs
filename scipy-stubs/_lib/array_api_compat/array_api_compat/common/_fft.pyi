from collections.abc import Sequence
from typing import Literal

from ._typing import Device as Device, ndarray as ndarray

def fft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def ifft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def fftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray: ...
def ifftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray: ...
def rfft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def irfft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def rfftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray: ...
def irfftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray: ...
def hfft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def ihfft(
    x: ndarray, /, xp, *, n: int | None = None, axis: int = -1, norm: Literal["backward", "ortho", "forward"] = "backward"
) -> ndarray: ...
def fftfreq(n: int, /, xp, *, d: float = 1.0, device: Device | None = None) -> ndarray: ...
def rfftfreq(n: int, /, xp, *, d: float = 1.0, device: Device | None = None) -> ndarray: ...
def fftshift(x: ndarray, /, xp, *, axes: int | Sequence[int] | None = None) -> ndarray: ...
def ifftshift(x: ndarray, /, xp, *, axes: int | Sequence[int] | None = None) -> ndarray: ...
