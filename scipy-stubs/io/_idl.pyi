from typing import Any

__all__ = ["readsav"]

class Pointer:
    index: int
    def __init__(self, index: int) -> None: ...

class ObjectPointer(Pointer): ...

class AttrDict(dict[str, Any]):
    def __init__(self, /, init: dict[str, Any] = ...) -> None: ...
    def __call__(self, name: str, /) -> Any: ...

def readsav(
    file_name: str,
    idict: dict[str, Any] | None = None,
    python_dict: bool = False,
    uncompressed_file_name: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]: ...
