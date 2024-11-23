from collections.abc import Mapping, Sequence
from typing import Literal, type_check_only

import numpy as np
import optype.numpy as onp
from ._constraints import Bounds, LinearConstraint
from ._optimize import OptimizeResult

@type_check_only
class _OptimizeResult(OptimizeResult):
    status: Literal[0, 1, 2, 3, 4]
    success: bool
    message: str
    x: onp.ArrayND[np.float64] | None
    fun: float | np.float64 | None
    mip_node_count: int | None
    mip_dual_bound: float | np.float64 | None
    mip_gap: float | np.float64 | None

###

def milp(
    c: onp.ToFloat1D,
    *,
    integrality: onp.ToInt1D | None = None,
    bounds: Bounds | None = None,
    constraints: Sequence[LinearConstraint] | None = None,
    options: Mapping[str, object] | None = None,  # TODO(jorenham): TypedDict
) -> _OptimizeResult: ...
