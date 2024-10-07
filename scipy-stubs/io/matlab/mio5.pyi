# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import Self, deprecated

__all__ = ["MatReadError", "MatReadWarning", "MatWriteError", "MatlabFunction", "MatlabObject", "mat_struct", "varmats_from_mat"]

@deprecated("will be removed in SciPy v2.0.0")
class MatReadError: ...

@deprecated("will be removed in SciPy v2.0.0")
class MatWriteError: ...

@deprecated("will be removed in SciPy v2.0.0")
class MatReadWarning: ...

@deprecated("will be removed in SciPy v2.0.0")
class MatlabFunction:
    def __new__(cls, input_array: object) -> Self: ...

@deprecated("will be removed in SciPy v2.0.0")
class MatlabObject:
    def __new__(cls, input_array: object, classname: object = ...) -> Self: ...
    def __array_finalize__(self, obj: Self) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class mat_struct: ...

@deprecated("will be removed in SciPy v2.0.0")
def varmats_from_mat(file_obj: object) -> object: ...
