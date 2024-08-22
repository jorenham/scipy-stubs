from ._base import issparse as issparse, sparray as sparray
from ._sputils import isintlike as isintlike
from scipy._typing import Untyped

INT_TYPES: Untyped

class IndexMixin:
    def __getitem__(self, key) -> Untyped: ...
    def __setitem__(self, key, x) -> None: ...
