__all__ = ["cascade", "cwt", "daub", "morlet", "morlet2", "qmf", "ricker"]

from collections.abc import Callable
from typing_extensions import deprecated

@deprecated("will be removed in SciPy 1.15")
def daub(p: int) -> object: ...
@deprecated("will be removed in SciPy 1.15")
def qmf(hk: object) -> object: ...
@deprecated("will be removed in SciPy 1.15")
def cascade(hk: object, J: int = 7) -> tuple[object, object, object]: ...
@deprecated("will be removed in SciPy 1.15")
def morlet(M: int, w: float = 5.0, s: float = 1.0, complete: bool = True) -> object: ...
@deprecated("will be removed in SciPy 1.15")
def ricker(points: int, a: object) -> object: ...
@deprecated("will be removed in SciPy 1.15")
def morlet2(M: int, s: float, w: float = 5) -> object: ...
@deprecated("will be removed in SciPy 1.15")
def cwt(
    data: object,
    wavelet: Callable[..., object],
    widths: object,
    dtype: object | None = None,
    **kwargs: object,
) -> object: ...
