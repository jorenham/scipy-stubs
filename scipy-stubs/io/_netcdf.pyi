from types import TracebackType

from scipy._typing import Untyped

IS_PYPY: Untyped
ABSENT: bytes
ZERO: bytes
NC_BYTE: bytes
NC_CHAR: bytes
NC_SHORT: bytes
NC_INT: bytes
NC_FLOAT: bytes
NC_DOUBLE: bytes
NC_DIMENSION: bytes
NC_VARIABLE: bytes
NC_ATTRIBUTE: bytes
FILL_BYTE: bytes
FILL_CHAR: bytes
FILL_SHORT: bytes
FILL_INT: bytes
FILL_FLOAT: bytes
FILL_DOUBLE: bytes
TYPEMAP: Untyped
FILLMAP: Untyped
REVERSE: Untyped

class netcdf_file:
    fp: Untyped
    filename: str
    use_mmap: Untyped
    mode: Untyped
    version_byte: Untyped
    maskandscale: Untyped
    dimensions: Untyped
    variables: Untyped
    def __init__(self, filename, mode: str = "r", mmap: Untyped | None = None, version: int = 1, maskandscale: bool = False): ...
    def __setattr__(self, attr, value) -> None: ...
    def close(self): ...
    __del__ = close
    def __enter__(self) -> Untyped: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None): ...
    def createDimension(self, name, length): ...
    def createVariable(self, name, type, dimensions) -> Untyped: ...
    def flush(self): ...
    sync = flush

class netcdf_variable:
    data: Untyped
    dimensions: Untyped
    maskandscale: Untyped
    @property
    def isrec(self) -> Untyped: ...
    @property
    def shape(self) -> Untyped: ...
    def __init__(
        self,
        data,
        typecode,
        size,
        shape,
        dimensions,
        attributes: Untyped | None = None,
        maskandscale: bool = False,
    ): ...
    def __setattr__(self, attr, value) -> None: ...
    def getValue(self) -> Untyped: ...
    def assignValue(self, value): ...
    def typecode(self) -> Untyped: ...
    def itemsize(self) -> Untyped: ...
    def __getitem__(self, index) -> Untyped: ...
    def __setitem__(self, index, data) -> None: ...

NetCDFFile = netcdf_file
NetCDFVariable = netcdf_variable
