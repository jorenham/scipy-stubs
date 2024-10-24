from scipy._typing import Untyped
from ._base import sparray as sparray
from ._coo import coo_array as coo_array, coo_matrix as coo_matrix

__docformat__: str

def find(A: Untyped) -> Untyped: ...
def tril(A: Untyped, k: int = 0, format: Untyped | None = None) -> Untyped: ...
def triu(A: Untyped, k: int = 0, format: Untyped | None = None) -> Untyped: ...
