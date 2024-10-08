# This module is not meant for public use and will be removed in SciPy v2.0.0.

from typing_extensions import deprecated

__all__ = ["ArffError", "MetaData", "ParseArffError", "loadarff"]

@deprecated("will be removed in SciPy v2.0.0")
class ArffError(OSError): ...

@deprecated("will be removed in SciPy v2.0.0")
class ParseArffError(ArffError): ...

@deprecated("will be removed in SciPy v2.0.0")
class MetaData:
    def __init__(self, /, rel: object, attr: object) -> None: ...
    def __getitem__(self, key: object, /) -> object: ...
    def __iter__(self, /) -> object: ...
    def names(self, /) -> object: ...
    def types(self, /) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
def loadarff(f: object) -> object: ...
