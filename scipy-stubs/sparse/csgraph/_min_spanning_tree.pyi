from typing import Final

import numpy as np
import optype.numpy as onp
from scipy.sparse import csr_matrix, sparray, spmatrix

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def minimum_spanning_tree(csgraph: onp.ToFloat2D | spmatrix | sparray, overwrite: bool = False) -> csr_matrix: ...
