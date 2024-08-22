from scipy._typing import Untyped
from scipy.sparse import (
    bsr_array as bsr_array,
    bsr_matrix as bsr_matrix,
    coo_array as coo_array,
    coo_matrix as coo_matrix,
    csc_array as csc_array,
    csc_matrix as csc_matrix,
    csr_array as csr_array,
    csr_matrix as csr_matrix,
)
from scipy.sparse._sputils import isscalarlike as isscalarlike

def toarray(a) -> Untyped: ...

formats_for_minmax: Untyped
formats_for_minmax_supporting_1d: Untyped

class Test_MinMaxMixin1D:
    def test_minmax(self, spcreator): ...
    def test_minmax_axis(self, spcreator): ...
    def test_numpy_minmax(self, spcreator): ...
    def test_argmax(self, spcreator): ...

class Test_ShapeMinMax2DWithAxis:
    def test_minmax(self, spcreator): ...
