from scipy._typing import Untyped
from scipy.sparse import issparse as issparse
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy as convert_pydata_sparse_to_scipy
from scipy.sparse.linalg import svds as svds

def norm(x, ord: Untyped | None = None, axis: Untyped | None = None) -> Untyped: ...
