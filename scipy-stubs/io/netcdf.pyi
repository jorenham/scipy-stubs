# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated, override

__all__ = ["netcdf_file", "netcdf_variable"]

@deprecated("will be removed in SciPy v2.0.0")
class netcdf_file:
    def __init__(
        self,
        filename: object,
        mode: object = ...,
        mmap: object = ...,
        version: object = ...,
        maskandscale: object = ...,
    ) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def createVariable(self, name: object, type: object, dimensions: object) -> object: ...
    def createDimension(self, name: object, length: object) -> None: ...
    def sync(self) -> None: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> object: ...
    def __exit__(self, type: object, value: object, traceback: object) -> None: ...
    @override
    def __setattr__(self, attr: object, value: object) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class netcdf_variable:
    def __init__(
        self,
        data: object,
        typecode: object,
        size: object,
        shape: object,
        dimensions: object,
        attributes: object = ...,
        maskandscale: object = ...,
    ) -> None: ...
    @property
    def isrec(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def assignValue(self, value: object) -> None: ...
    def getValue(self) -> object: ...
    def itemsize(self) -> object: ...
    def typecode(self) -> object: ...
    def __getitem__(self, index: object) -> object: ...
    @override
    def __setattr__(self, attr: object, value: object) -> None: ...
    def __setitem__(self, index: object, data: object) -> None: ...
