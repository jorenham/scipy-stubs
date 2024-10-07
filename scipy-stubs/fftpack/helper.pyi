# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["fftfreq", "fftshift", "ifftshift", "next_fast_len", "rfftfreq"]

@deprecated("will be removed in SciPy v2.0.0")
def fftfreq(
    n: object,
    d: object = ...,
    device: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def fftshift(x: object, axes: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def ifftshift(x: object, axes: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def next_fast_len(target: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def rfftfreq(n: object, d: object = ...) -> object: ...
