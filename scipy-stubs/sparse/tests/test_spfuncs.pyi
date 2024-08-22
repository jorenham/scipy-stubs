from scipy._typing import Untyped
from scipy.sparse import bsr_matrix as bsr_matrix, csc_matrix as csc_matrix, csr_matrix as csr_matrix
from scipy.sparse._sparsetools import (
    bsr_scale_columns as bsr_scale_columns,
    bsr_scale_rows as bsr_scale_rows,
    csr_scale_columns as csr_scale_columns,
    csr_scale_rows as csr_scale_rows,
)

class TestSparseFunctions:
    def test_scale_rows_and_cols(self): ...
    def test_estimate_blocksize(self): ...
    def test_count_blocks(self) -> Untyped: ...
