import io
from typing_extensions import override

from scipy._typing import Untyped

__all__ = ["mminfo", "mmread", "mmwrite"]

PARALLELISM: int
ALWAYS_FIND_SYMMETRY: bool

class _TextToBytesWrapper(io.BufferedReader):
    encoding: Untyped
    errors: Untyped
    def __init__(self, text_io_buffer, encoding: Untyped | None = None, errors: Untyped | None = None, **kwargs): ...
    @override
    def seek(self, offset: int, whence: int = 0, /) -> None: ...  # type: ignore[override]

def mmread(source) -> Untyped: ...
def mmwrite(
    target,
    a,
    comment: Untyped | None = None,
    field: Untyped | None = None,
    precision: Untyped | None = None,
    symmetry: str = "AUTO",
): ...
def mminfo(source) -> Untyped: ...
