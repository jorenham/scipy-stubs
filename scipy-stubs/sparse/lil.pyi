# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["isspmatrix_lil", "lil_array", "lil_matrix"]

@deprecated("will be removed in SciPy v2.0.0")
class lil_array:
    def __init__(
        self,
        arg1: object,
        shape: object = ...,
        dtype: object = ...,
        copy: object = ...,
    ) -> None: ...
    def getrowview(self, i: object) -> object: ...
    def getrow(self, i: object) -> object: ...
    def resize(self, *shape: int) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class lil_matrix: ...

@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_lil(x: object) -> object: ...
