from ._base import issparse as issparse
from ._csr import csr_array as csr_array
from ._sparsetools import csr_count_blocks as csr_count_blocks
from scipy._typing import Untyped

def estimate_blocksize(A, efficiency: float = 0.7) -> Untyped: ...
def count_blocks(A, blocksize) -> Untyped: ...
