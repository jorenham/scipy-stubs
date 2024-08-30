from typing import Final

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped

DTYPE: Final = np.float64

def validate_graph(
    csgraph: Untyped,
    directed: Untyped,
    dtype: npt.DTypeLike = ...,
    csr_output: bool = True,
    dense_output: bool = True,
    copy_if_dense: bool = False,
    copy_if_sparse: bool = False,
    null_value_in: int = 0,
    null_value_out: Untyped = ...,
    infinity_null: bool = True,
    nan_null: bool = True,
) -> Untyped: ...
