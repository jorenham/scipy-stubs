from typing import Final, Literal

import numpy as np
from scipy.sparse import csr_matrix

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

class MaximumFlowResult:
    flow_value: Final[int | np.int32 | np.int64]
    flow: Final[csr_matrix]

    def __init__(self, /, flow_value: int, flow: csr_matrix) -> None: ...

def maximum_flow(
    csgraph: csr_matrix,
    source: int,
    sink: int,
    *,
    method: Literal["edmonds_karp", "dinic"] = "dinic",
) -> MaximumFlowResult: ...
