from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, sparray, spmatrix

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def minimum_spanning_tree(csgraph: spmatrix | sparray | npt.ArrayLike, overwrite: bool = False) -> csr_matrix: ...
