from scipy._typing import Untyped
from scipy.io import hb_read as hb_read, hb_write as hb_write
from scipy.sparse import coo_matrix as coo_matrix, csc_matrix as csc_matrix, rand as rand

SIMPLE: str
SIMPLE_MATRIX: Untyped

def assert_csc_almost_equal(r, l): ...

class TestHBReader:
    def test_simple(self): ...

class TestHBReadWrite:
    def check_save_load(self, value): ...
    def test_simple(self): ...
