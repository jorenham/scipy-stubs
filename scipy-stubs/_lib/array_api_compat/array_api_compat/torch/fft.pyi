from collections.abc import Sequence
from typing import Literal

from torch.fft import *

from scipy._typing import Untyped

array: Untyped

def fftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array: ...
def ifftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array: ...
def rfftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array: ...
def irfftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array: ...
def fftshift(x: array, /, *, axes: int | Sequence[int] | None = None, **kwargs) -> array: ...
def ifftshift(x: array, /, *, axes: int | Sequence[int] | None = None, **kwargs) -> array: ...
