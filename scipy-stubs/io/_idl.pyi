from scipy._typing import Untyped

DTYPE_DICT: Untyped
RECTYPE_DICT: Untyped
STRUCT_DICT: Untyped

class Pointer:
    index: Untyped
    def __init__(self, index) -> None: ...

class ObjectPointer(Pointer): ...

class AttrDict(dict):
    def __init__(self, init: Untyped | None = None): ...
    def __getitem__(self, name) -> Untyped: ...
    def __setitem__(self, key, value) -> None: ...
    def __getattr__(self, name) -> Untyped: ...
    __setattr__ = __setitem__
    __call__ = __getitem__

def readsav(
    file_name,
    idict: Untyped | None = None,
    python_dict: bool = False,
    uncompressed_file_name: Untyped | None = None,
    verbose: bool = False,
) -> Untyped: ...
