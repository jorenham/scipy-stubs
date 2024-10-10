# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing import Final
from typing_extensions import deprecated

__all__ = ["Model", "exponential", "multilinear", "polynomial", "quadratic", "unilinear"]

@deprecated("will be removed in SciPy v2.0.0")
class Model:
    def __init__(
        self,
        fcn: object,
        fjacb: object = ...,
        fjacd: object = ...,
        extra_args: object = ...,
        estimate: object = ...,
        implicit: object = ...,
        meta: object = ...,
    ) -> None: ...
    def set_meta(self, **kwds: object) -> None: ...
    def __getattr__(self, attr: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
def polynomial(order: object) -> object: ...

multilinear: Final[object]
exponential: Final[object]
unilinear: Final[object]
quadratic: Final[object]
