# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["Rbf"]

@deprecated("will be removed in SciPy v2.0.0")
class Rbf:
    @property
    def A(self) -> object: ...
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __call__(self, *args: object) -> object: ...
