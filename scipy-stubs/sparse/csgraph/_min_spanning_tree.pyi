from typing import Final, TypeAlias

import numpy as np
import optype.numpy as onp
from scipy.sparse import csr_matrix
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Floating, Integer

_Real: TypeAlias = Integer | Floating
_ToGraph: TypeAlias = onp.ToFloat2D | _spbase[_Real, tuple[int, int]]

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def minimum_spanning_tree(csgraph: onp.ToFloat2D | _ToGraph, overwrite: bool = False) -> csr_matrix[_Real]: ...
